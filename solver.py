import time
from utils.utils import *
from model.MambaAnomalyTransformer import MambaAnomalyTransformer
from data_factory.data_loader import get_loader_segment
from metrics.Matthews_correlation_coefficient import *
from metrics.metrics import combine_all_evaluation_scores


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def sliding_quantile_threshold(scores, window_size=200, quantile=0.995):
    """
    For each index t, threshold[t] = quantile of scores in the last window_size values
    ending at t (past and current only; shorter prefix when t < window_size - 1).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(scores)
    thresholds = np.empty(n, dtype=np.float64)
    for t in range(n):
        start = max(0, t - window_size + 1)
        thresholds[t] = np.quantile(scores[start : t + 1], quantile)
    return thresholds


def phase_bucket_threshold(scores, win_size, num_buckets=10, quantile=0.995, min_count=20):
    """
    Minimal phase-aware threshold: phase_idx = t % win_size, bucket over [0, win_size-1] -> num_buckets.
    Causal only; threshold[t] from same-bucket history including t, else global quantile on scores[0:t+1].
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(scores)
    ws = max(int(win_size), 1)
    nb = max(int(num_buckets), 1)
    thresholds = np.empty(n, dtype=np.float64)
    bucket_scores = [[] for _ in range(nb)]

    def bucket_id(t):
        phase_idx = t % ws
        b = int(phase_idx * nb / float(ws))
        return min(max(b, 0), nb - 1)

    for t in range(n):
        b = bucket_id(t)
        bucket_scores[b].append(scores[t])
        same = bucket_scores[b]
        if len(same) >= min_count:
            thresholds[t] = np.quantile(np.asarray(same, dtype=np.float64), quantile)
        else:
            thresholds[t] = np.quantile(scores[0 : t + 1], quantile)
    return thresholds


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        gpu_index = self.get_gpu_index()
        self.device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def get_gpu_index(self):
        gpu_index = self.gpu_index
        return gpu_index

    def build_model(self):
        self.model = MambaAnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.get_gpu_index())
            self.model.cuda()


    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        rec_losses = []  # List to store reconstruction losses

        for i, batch in enumerate(vali_loader):
            # Version A dataloader returns (x, c, label); keep backward-compat with (x, label).
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input_data, context_data, _ = batch
            else:
                input_data, _ = batch
                context_data = None

            input = input_data.float().to(self.device)
            context = context_data.float().to(self.device) if context_data is not None else None
            output, series, prior, _ = self.model(input, context)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)

            # Store reconstruction loss for plotting
            rec_losses.append(rec_loss.item())

            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())
        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                # Version A dataloader returns (x, c, label); keep backward-compat with (x, label).
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    input_data, context_data, labels = batch
                else:
                    input_data, labels = batch
                    context_data = None

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                context = context_data.float().to(self.device) if context_data is not None else None

                output, series, prior, _ = self.model(input, context)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)
                #print(rec_loss.item())
                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")
        use_adapt = getattr(self, 'use_adaptive_threshold', False)
        tw = getattr(self, 'threshold_window', 200)
        tq = getattr(self, 'threshold_quantile', 0.995)
        thr_mode = getattr(self, 'threshold_mode', 'fixed')
        # Legacy: use_adaptive_threshold=True with default mode fixed -> sliding
        if thr_mode == 'fixed' and use_adapt:
            thr_mode = 'sliding'
        print("threshold_mode: {} | legacy use_adaptive_threshold: {}".format(thr_mode, use_adapt))
        print("threshold_window: {}, threshold_quantile: {}".format(tw, tq))

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, batch in enumerate(self.train_loader):
            # Version A dataloader returns (x, c, label); keep backward-compat with (x, label).
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input_data, context_data, labels = batch
            else:
                input_data, labels = batch
                context_data = None

            input = input_data.float().to(self.device)
            context = context_data.float().to(self.device) if context_data is not None else None
            output, series, prior, _ = self.model(input, context)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            # Version A dataloader returns (x, c, label); keep backward-compat with (x, label).
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input_data, context_data, labels = batch
            else:
                input_data, labels = batch
                context_data = None

            input = input_data.float().to(self.device)
            context = context_data.float().to(self.device) if context_data is not None else None
            output, series, prior, _ = self.model(input, context)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            # Version A dataloader returns (x, c, label); keep backward-compat with (x, label).
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input_data, context_data, labels = batch
            else:
                input_data, labels = batch
                context_data = None

            input = input_data.float().to(self.device)
            context = context_data.float().to(self.device) if context_data is not None else None
            output, series, prior, _ = self.model(input, context)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        # Per-mode pred: fixed = scalar thresh; sliding / phase_bucket = per-t thresholds
        if thr_mode == 'sliding':
            thresholds = sliding_quantile_threshold(test_energy, window_size=tw, quantile=tq)
            print("[sliding] thresholds mean={:.6f}, head={}".format(
                float(np.mean(thresholds)), thresholds[: min(5, len(thresholds))]))
            pred = (test_energy > thresholds).astype(int)
        elif thr_mode == 'phase_bucket':
            # Params from config (main.py): phase_num_buckets, threshold_quantile, phase_min_count
            nb = getattr(self, 'phase_num_buckets', 10)
            ph_q = getattr(self, 'threshold_quantile', 0.995)
            ph_min = getattr(self, 'phase_min_count', 20)
            print("[phase_bucket] win_size={}, num_buckets={}, quantile={}, min_count={}".format(
                self.win_size, nb, ph_q, ph_min))
            thresholds = phase_bucket_threshold(
                test_energy,
                win_size=self.win_size,
                num_buckets=nb,
                quantile=ph_q,
                min_count=ph_min,
            )
            print("[phase_bucket] thresholds mean={:.6f}, head={}".format(
                float(np.mean(thresholds)), thresholds[: min(5, len(thresholds))]))
            pred = (test_energy > thresholds).astype(int)
        else:
            # fixed: global percentile on combined_energy (train + thre_loader stats)
            pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        matrix = [137]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.2f}, Precision : {:0.2f}, Recall : {:0.2f}, F-score : {:0.2f} ".format(
                round(accuracy * 100, 2), round(precision * 100, 2),
                round(recall * 100, 2), round(f_score * 100, 2)))

        return accuracy, precision, recall, f_score


# Example usage
if __name__ == '__main__':
    gt = np.load("data/events_pred_MSL.npy") + 0
    pred = np.load("data/events_gt_MSL.npy") + 0
    anomaly_scores = np.load("data/events_scores_MSL.npy")
