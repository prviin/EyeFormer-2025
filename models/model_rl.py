from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import torch


def DTW(P, Q):
    '''
    Compute Dynamic Time Warping distance between two Numpy arrays.
    '''
    dist, _ = fastdtw(P, Q, dist=euclidean)
    return dist


def discount_rewards(r, gamma=0.99):
    """ Compute the gamma-discounted rewards over an episode
    """
    discounted_r, cumul_r = np.zeros_like(r), 0
    for t in reversed(range(0, len(r))):
        cumul_r = r[t] + cumul_r * gamma
        discounted_r[t] = cumul_r
    return discounted_r


def create_circular_mask(h, w, fixations_x, fixations_y, r1, r2):

    bs = fixations_x.shape[0]
    # get the circular mask
    mask = torch.zeros(bs, h, w)
    X, Y = np.ogrid[:h, :w]
    X = X[np.newaxis, ...].repeat(bs, axis=0)
    Y = Y[np.newaxis, ...].repeat(bs, axis=0)
    r1 = r1.data.cpu().numpy()[:, np.newaxis, np.newaxis]
    r2 = r2.data.cpu().numpy()[:, np.newaxis, np.newaxis]
    for i in range(fixations_x.shape[1]):
        ### The formula of ellipse
        dist = np.sqrt((X - fixations_x[:, i:i+1][:,:,np.newaxis])**2 / (r2**2) + (Y - fixations_y[:, i:i+1][:,:,np.newaxis])**2 / (r1**2))
        mask = torch.maximum(mask, torch.from_numpy(dist <= 1))

    return 1 - mask


def process_saliency(res, saliency, w, h):
    saliency_height = saliency.shape[1]
    saliency_width = saliency.shape[2]

    w = w.float()
    h = h.float()
    scalar = torch.minimum(1920.0 / w, 1200.0 / h)

    # the radius should be one visual angle divided by the resize ratio.
    # We pre-set a radius value of 120
    radius = 120.0 / scalar
    r1 = radius / w * saliency_width
    r2 = radius / h * saliency_height

    row_indices = np.clip(
        np.rint(res[:, :, 0] * (saliency_height - 1)), 0, saliency_height - 1
    ).astype(int)
    col_indices = np.clip(
        np.rint(res[:, :, 1] * (saliency_width - 1)), 0, saliency_width - 1
    ).astype(int)

    seq_len = row_indices.shape[1]
    saliency_value_list = []
    for i in range(seq_len):
        res_pos_range = np.arange(row_indices.shape[0])[:, np.newaxis]
        res_pos_row = row_indices[:, i][:, np.newaxis]
        res_pos_col = col_indices[:, i][:, np.newaxis]
        saliency_value = saliency[res_pos_range, res_pos_row, res_pos_col]
        saliency_value_list.append(saliency_value)

        fixations_x = row_indices[:, :i+1]
        fixations_y = col_indices[:, :i+1]

        ### Block the surrounding areas of the predicted points
        saliency_mask = create_circular_mask(saliency_height, saliency_width, fixations_x, fixations_y, r1, r2)
        saliency_mask = saliency_mask.data.cpu().numpy()
        saliency = saliency * saliency_mask
    saliency_value_list = np.concatenate(saliency_value_list, 1)
    return saliency_value_list


def get_self_critical_reward(greedy_pred, data_gts, time_gts, gen_pred, saliency_image, width, height):
    batch_size = len(data_gts)
    assert greedy_pred.shape[0] == batch_size

    greedy_res = greedy_pred.data.cpu().numpy()
    gen_result = gen_pred.data.cpu().numpy()
    saliency_image = saliency_image.cpu().numpy()

    ### Get the saliency reward for all the steps, the duration is excluded
    greedy_saliency = process_saliency(greedy_res[:, :, :2], saliency_image, width, height)
    gen_saliency = process_saliency(gen_result[:, :, :2], saliency_image, width, height)

    gen_score_list = []
    greedy_score_list = []

    for i, (greedy_r, gen_r, gts_r, gts_t_r) in enumerate(zip(greedy_res, gen_result, data_gts, time_gts)):
        gts_r = [np.array(e).astype(np.float32) for e in gts_r]
        gts_t_r = [np.array(e).astype(np.float32)[:, np.newaxis] for e in gts_t_r]

        gts_r = [np.concatenate([e1, e2], 1) for e1, e2 in zip(gts_r, gts_t_r)]

        ### Get the DTW reward. As there are multiple scanpaths, we calculate the mean DTW value
        gen_dtw = [DTW(gen_r, e) for e in gts_r]
        gen_dtw = sum(gen_dtw) / len(gen_dtw)

        ### We use negative dtw value to make sure that dtw is encouraged to decrease.
        ### NOTE: discounted rewards are used for saliency values
        gen_score = -gen_dtw + discount_rewards(gen_saliency[i])

        greedy_dtw = [DTW(greedy_r, e) for e in gts_r]
        greedy_dtw = sum(greedy_dtw) / len(greedy_dtw)
        greedy_score = -greedy_dtw + discount_rewards(greedy_saliency[i])
        gen_score_list.append(gen_score)
        greedy_score_list.append(greedy_score)

    gen_score_list = np.array(gen_score_list)
    greedy_score_list = np.array(greedy_score_list)

    ### The difference btw the reward and baseline
    rewards = gen_score_list - greedy_score_list
    rewards = rewards[:, :, np.newaxis]
    return rewards
