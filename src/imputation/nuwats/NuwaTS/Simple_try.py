import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch

from exp.exp_imputation import Exp_Imputation

import torch.nn as nn


def plot_sin(x_len=96):
    x = np.linspace(0, 4 * np.pi, x_len)
    y = (np.sin(x) + 1) / 2
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(0, 1)
    plt.show()


def plot_cos(x_len=96):
    x = np.linspace(0, 4 * np.pi, x_len)
    y = (np.cos(x) + 1) / 2
    plt.plot(x, y)
    plt.title("Cosine Wave")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(0, 1)
    plt.show()


def plot_exp(x_len=96):
    x = np.linspace(0, 4, x_len)
    y = np.exp(x) / np.exp(4)
    plt.plot(x, y)
    plt.title("Exponential Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(0, 1)
    plt.show()


def plot_softmax(x_len=96):
    x = np.linspace(0, 4, x_len)
    y = np.exp(x) / np.sum(np.exp(x))
    plt.plot(x, y)
    plt.title("Softmax Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(0, 1)
    plt.show()


def plot_tanh(x_len=96):
    x = np.linspace(-4, 4, x_len)
    y = (np.tanh(x) + 1) / 2
    plt.plot(x, y)
    plt.title("Hyperbolic Tangent Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(0, 1)
    plt.show()


def plot_relu(x_len=96):
    x = np.linspace(-4, 4, x_len)
    y = np.maximum(x, 0) / 4
    plt.plot(x, y)
    plt.title("ReLU Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(0, 1)
    plt.show()


def plot_all(x_len=96):
    x = np.linspace(0, 4 * np.pi, x_len)
    y_sin = (np.sin(x) + 1) / 2
    y_cos = (np.cos(x) + 1) / 2
    # y_exp = np.exp(x) / np.exp(4)

    # sigmoid tanh softmax
    x = np.linspace(-6, 6, x_len)
    y_sigmoid = 1 / (1 + np.exp(-x))
    y_tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # c = np.max(x)  # 解决溢出问题
    exe_x = np.exp(x)
    exe_s = np.sum(exe_x)
    y_softmax = exe_x / exe_s

    a = np.random.rand()
    b = np.random.rand()
    return (
        a * y_sin + b,
        a * y_cos + b,
        a * y_softmax + b,
        a * y_tanh + b,
        a * y_sigmoid + b,
        a,
        b,
    )


def generate_mask(batch_x, ratio=0.1):
    batch_x = torch.from_numpy(batch_x).float().to("cuda:0")
    batch_x = batch_x.unsqueeze(0).unsqueeze(-1)
    B, T, N = batch_x.shape
    mask = torch.rand((B, T, N)).to("cuda:0")
    random_mask_rate = ratio
    num_masked = int(T * random_mask_rate)
    shuffle_indices = torch.rand(B, T, N, device="cuda:0").argsort(1)
    mask_ind, unmask_ind = (
        shuffle_indices[:, :num_masked, :],
        shuffle_indices[:, num_masked:, :],
    )
    batch_ind = torch.arange(B, device="cuda:0").unsqueeze(-1).unsqueeze(-1)
    sensor_ind = torch.arange(N, device="cuda:0").unsqueeze(0).unsqueeze(0)
    mask[batch_ind, mask_ind, sensor_ind] = 0  # masked
    mask[batch_ind, unmask_ind, sensor_ind] = 1  # remained

    inp = batch_x.masked_fill(mask == 0, 0)
    return batch_x, inp, mask


def visual(true, preds=None, name="./pic/test.pdf", mask_rate=0.8, a=1, b=1):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.title("mask_rate:{}, a:{}, b:{}".format(mask_rate, a, b))
    plt.savefig(name, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NuwaTS")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        default="denoise",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, denoise]",
    )
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        default="NuwaTS",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # data loader
    parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )

    # imputation task
    parser.add_argument(
        "--test_mask_rate", type=float, default=0.8, help="test mask ratio"
    )
    parser.add_argument("--max_iterations", type=int, default=10, help="max iterations")
    parser.add_argument(
        "--max_optimization_iterations",
        type=int,
        default=10,
        help="max optimization iterations",
    )
    parser.add_argument(
        "--regularization_weight",
        type=float,
        default=0.05,
        help="regularization weight",
    )

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)"
    )

    # model define
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )
    # patching
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gpt_layers", type=int, default=6)
    parser.add_argument("--ln", type=int, default=0)
    parser.add_argument("--mlp", type=int, default=0)
    parser.add_argument("--weight", type=float, default=0)
    parser.add_argument("--percent", type=int, default=5)

    # prefix tuning
    parser.add_argument("--prefix_tuning", action="store_true", help="", default=True)
    parser.add_argument(
        "--prefix_tuningv2", action="store_true", help="", default=False
    )
    parser.add_argument(
        "--continue_tuning", action="store_true", help="", default=False
    )
    parser.add_argument(
        "--continue_tuningv2", action="store_true", help="", default=False
    )

    parser.add_argument("--frozen_lm", action="store_true", help="", default=False)
    parser.add_argument("--prefix_length", type=int, default=6)
    parser.add_argument("--train_all_lm", action="store_true", help="", default=False)
    parser.add_argument("--use_llama", action="store_true", help="", default=False)
    parser.add_argument("--alignment", action="store_true", help="", default=False)

    # contrastive
    parser.add_argument("--con_weight", type=float, default=0.01, help="")
    parser.add_argument("--patch_con", action="store_true", help="", default=False)
    parser.add_argument("--temporal_con", action="store_true", help="", default=False)
    parser.add_argument("--flatten_con", action="store_true", help="", default=False)
    parser.add_argument("--best_con_num", type=int, default=128)
    # output learnable token
    parser.add_argument("--seq_token", type=int, default=0)
    parser.add_argument("--word_prompt", action="store_true", help="", default=False)
    parser.add_argument("--cov_prompt", action="store_true", help="", default=True)
    parser.add_argument("--output_token", action="store_true", help="", default=False)

    # test
    parser.add_argument("--test_all", action="store_true", help="", default=False)

    # forecasting
    parser.add_argument("--is_forecasting", action="store_true", help="", default=False)
    parser.add_argument(
        "--auto_regressive", action="store_true", help="", default=False
    )

    args = parser.parse_args()

    Exp = Exp_Imputation
    exp = Exp(args)
    Ours = exp.model
    Path = "mathmatic_10000.pth"
    Ours.load_state_dict(torch.load(Path, map_location="cuda:0"))

    folder_path = "finetune_math10000/"
    if not args.is_training:
        for mask_rate in range(1, 10):
            mask_rate = mask_rate / 10
            input_list = plot_all()
            a = input_list[-2]
            b = input_list[-1]
            input_list = input_list[:-2]
            for i, batch_x in enumerate(input_list):
                with torch.no_grad():
                    batch_x, inp, mask = generate_mask(batch_x, ratio=mask_rate)
                    outputs, _ = Ours(inp, None, None, None, mask)
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                filled = true[0, :, -1].copy()
                filled_pred = filled * mask[0, :, -1] + pred[0, :, -1] * (
                    1 - mask[0, :, -1]
                )
                # visual(true[0, :, -1], filled_pred, os.path.join(folder_path, str(i) + 'rate{}.pdf'.format(mask_rate)),
                #        mask_rate=mask_rate,a=a,b=b)
                visual(
                    true[0, :, -1],
                    pred[0, :, -1],
                    os.path.join(
                        folder_path, str(i) + "origin_rate{}.pdf".format(mask_rate)
                    ),
                    mask_rate=mask_rate,
                    a=a,
                    b=b,
                )

    else:
        training_step = 10000
        criterion = nn.MSELoss()
        if (
            exp.args.prefix_tuningv2
            or exp.args.prefix_tuning
            or exp.args.continue_tuningv2
            or exp.args.continue_tuning
        ):
            print("load LargeST model or Fused General pretraining model")
            Path = "/usr/local/Wyk_team/Chengjinguo/LLM4TS/LLM4TS/checkpoints/bs512_prefixv1_1/0.020180checkpoint.pth"
            exp.model.load_state_dict(
                torch.load(Path, map_location=exp.device), strict=False
            )
            for i, (name, param) in enumerate(exp.model.named_parameters()):
                if "prefix" in name:  # or 'mlp' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        model_optim = exp._select_optimizer()
        for step in range(training_step):
            mask_rate = torch.rand(1).item() * 0.8 + 0.1
            math_list = plot_all()
            a = math_list[-2]
            b = math_list[-1]
            math_list = math_list[:-2]
            input_list = []
            mask_list = []
            ground_truth = []
            for i, batch_x in enumerate(math_list):
                batch_x_torch, inp, mask = generate_mask(batch_x, ratio=mask_rate)
                input_list.append(inp)
                mask_list.append(mask)
                ground_truth.append(batch_x_torch)
            input_list = torch.concatenate(input_list, dim=0)
            mask_list = torch.concatenate(mask_list, dim=0)
            ground_truth = torch.concatenate(ground_truth, dim=0)
            outputs, _ = Ours(input_list, None, None, None, mask_list)
            loss = criterion(outputs, ground_truth)
            loss.backward()
            model_optim.step()
            if step % 100 == 0:
                print("training_steps: {}".format(step))
        torch.save(Ours.state_dict(), "mathmatic_{}.pth".format(training_step))
        # outputs = outputs.detach().cpu().numpy()
        # pred = outputs
        # true = batch_x.detach().cpu().numpy()
        # mask = mask.detach().cpu().numpy()
        # filled = true[0, :, -1].copy()
        # filled_pred = filled * mask[0, :, -1] + \
        #               pred[0, :, -1] * (1 - mask[0, :, -1])
        # # visual(true[0, :, -1], filled_pred, os.path.join(folder_path, str(i) + 'rate{}.pdf'.format(mask_rate)),
        # #        mask_rate=mask_rate,a=a,b=b)
        # visual(true[0, :, -1], pred[0, :, -1],
        #        os.path.join(folder_path, str(i) + 'origin_rate{}.pdf'.format(mask_rate)), mask_rate=mask_rate,a=a,b=b)
