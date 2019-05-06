import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable

def instance_bce_with_logits(logits, labels, not_yn=None):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    if not_yn is not None:
        loss *= not_yn
    loss = loss.mean()
    loss *= labels.size(1)
    return loss

# def instance_bce_with_logits(logits, labels):
#     assert logits.dim() == 2

#     loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
#     loss *= labels.size(1)
#     return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def compute_score_r(logits, labels):
    logits_idx = torch.argmax(logits, dim=1, keepdim=True)
    # print("compute_score_r: logits_idx", logits_idx.size(), "labels", labels.size())
    eqs = (logits_idx == labels).float()
    # print("compute_score_r:eqs", eqs.size())
    return eqs

def train(model, train_loader, eval_loader, num_epochs, output, not_yn=None):
    print("starting training")
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    bcelos = torch.nn.BCEWithLogitsLoss(reduction='none')
    best_eval_score = 0
    loss_dist_fn = nn.MSELoss(reduction='none')
    num_repeat = 4
    # eval_score, bound = evaluate(model, eval_loader)

    itr = 0
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        model.train(False)
        eval_score, bound, q_score, yn_score_eval = evaluate(
            model, eval_loader)
        logger.writer.add_scalar('score/eval', eval_score.item(), itr)
        logger.writer.add_scalar('score/q_eval', q_score.item(), itr)
        logger.writer.add_scalar('score/yn_eval', yn_score_eval, itr)
        model.train(True)
        # exit()

        for i, (v, b, q, a, word_vec, q_type) in enumerate(train_loader):
            for repeat_i in range(num_repeat):
                itr += 1
                optim.step()
                # print("v.size", v.size(), "v.size(0)", v.size(0), "itr", itr)
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                word_vec = word_vec.cuda()
                q_type = q_type.cuda()
                not_yn = (q_type > 1).float()
                yn = 1-not_yn
                yn_ans = (q_type == 1).float()
                print("qtype", q_type.size(), 'not_yn', not_yn.size())
                # print("a = " , a)
                # print("q = " , q)
                pred, pred_word_vec, pred_qtype, pred_yn = model(v, b, q, a)
                # print("pred_word_vec size = ", pred_word_vec.size())
                # print("word_vec size = ", word_vec.size())
                # print("pred_qtype", pred_qtype.size(), "qtype", q_type.size())
                loss_qtype = bcelos(pred_qtype, not_yn).mean() * (10**1)
                loss_bce = instance_bce_with_logits(pred, a, not_yn)

                loss_dist = (loss_dist_fn(pred_word_vec, word_vec)*not_yn).mean() * 10
                loss_yn = (bcelos(pred_yn, yn_ans)*(yn)).mean() * 50
                print("loss_yn", loss_yn.size())
                
                # loss_dist = 0
                loss = loss_bce + loss_dist + loss_qtype +loss_yn
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                batch_score = compute_score_with_logits(pred, a.data).sum()
                # qf_type = q_type.float()
                qt_score = compute_score_r(pred_qtype, q_type).mean()
                ynsum = yn.sum()
                if ynsum > 0:
                    yn_score = (compute_score_r(pred_yn, yn_ans.long()) * yn).sum()# * (pred_yn.size(0) / yn.sum())
                else:
                    yn_score = 0
                print("foogazi")
                # print("loss_bce[{}]\tloss_dist[{}]\tloss_qtype[{}]\tbatch_score[{}]\tqtscore[{}]".format(9, 9, 12, 10, 11))
                print("loss_bce[{:.4f}]\tloss_dist[{:.4f}]\tloss_qtype[{:.4f}]\tbatch_score[{:.5f}]\tynscore[{:.4f}]\tynloss[{}]\tynloss[{:.6f}]".format(loss_bce.item(), loss_dist.item(), loss_qtype.item(), batch_score.item(), qt_score.item(), yn_score.item(), loss_yn.item()))
                # print("loss", loss.item())
                loss_curr = loss.item() * v.size(0)
                logger.writer.add_scalar('loss/train', loss_curr, itr)
                logger.writer.add_scalar('score/train', batch_score.item(), itr)
                logger.writer.add_scalar('score/qtrain', qt_score.item(), itr)
                total_loss += loss_curr
                train_score += batch_score
                print('epoch %d: [%d(%d)]-> \ttrain_loss: %.5f' % (epoch, itr, i, loss_curr))
            # break
            

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        # model.train(False)
        # eval_score, bound, q_score, yn_score_eval = evaluate(model, eval_loader)
        # logger.writer.add_scalar('score/eval', eval_score.item(), itr)
        # logger.writer.add_scalar('score/q_eval', q_score.item(), itr)
        # logger.writer.add_scalar('score/yn_eval', yn_score_eval, itr)
        # model.train(True)

        # print('epoch [{}]-> \ttrain_loss: {}, score: {} \teval score: {} ({}) \teval_qscore \t qeval_score {}'.format(epoch, total_loss, train_score, 100 * eval_score, 100 * bound, q_score))
        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        # logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
        # logger.write('\tq_eval_score: %0.2f' % q_score)

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    q_score = 0
    yn_pscore = 0
    for v, b, q, a, word_vec, q_type in iter(dataloader):
        # v = Variable(v, volatile=True).cuda()
        # b = Variable(b, volatile=True).cuda()
        # q = Variable(q, volatile=True).cuda()
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()
        pred, pred_word_vec, pred_qtype, pred_yn = model(v, b, q, None)
        q_type = q_type.cuda()
        not_yn = (q_type > 1).float()
        yn = 1-not_yn
        yn_ans = (q_type == 1).float()
        batch_score = (compute_score_with_logits(pred, a.cuda())*not_yn).sum()
        yn_score = (compute_score_r(pred_yn, yn_ans.long())*yn).sum()
        batch_score += yn_score
        q_type = q_type.cuda()
        score += batch_score
        qt_score = compute_score_r(pred_qtype, q_type).mean() * q_type.size(0)
        q_score += qt_score
        yn_pscore += yn_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        # break # rm it

    score = score / len(dataloader.dataset)
    q_score = q_score / len(dataloader.dataset)
    yn_pscore = yn_pscore / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound, q_score, yn_pscore
