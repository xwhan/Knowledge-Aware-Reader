import torch
import numpy as np
import random

from data_generator import DataLoader
from model import KAReader
from util import get_config, cal_accuracy, load_documents

from tensorboardX import SummaryWriter

def f1_and_hits(answers, candidate2prob, eps):
    retrieved = []
    correct = 0
    best_ans, max_prob = -1, 0
    for c, prob in candidate2prob.items():
        if prob > max_prob:
            max_prob = prob
            best_ans = c
        if prob > eps:
            retrieved.append(c)
            if c in answers:
                correct += 1
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0
        else:
            return 0.0, 1.0
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 0.0, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return f1, hits

def get_best_ans(candidate2prob):
    best_ans, max_prob = -1, 0
    for c, prob in candidate2prob.items():
        if prob > max_prob:
            max_prob = prob
            best_ans = c
    return best_ans

def train(cfg):
    tf_logger = SummaryWriter('tf_logs/' + cfg['model_id'])

    # train and test share the same set of documents
    documents = load_documents(cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])

    # train data
    train_data = DataLoader(cfg, documents)
    valid_data = DataLoader(cfg, documents, mode='dev')

    model = KAReader(cfg)
    model = model.to(torch.device('cuda'))

    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(trainable, lr=cfg['learning_rate'])

    if cfg['lr_schedule']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [30], gamma=0.5)

    model.train()
    best_val_f1 = 0
    best_val_hits = 0
    for epoch in range(cfg['num_epoch']):
        batcher = train_data.batcher(shuffle=True)
        train_loss = []
        for feed in batcher:
            loss, pred, pred_dist = model(feed)
            train_loss.append(loss.item())
            # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
            # train_acc.append(acc)
            # train_max_acc.append(max_acc)
            optim.zero_grad()
            loss.backward()
            if cfg['gradient_clip'] != 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg['gradient_clip'])
            optim.step()
        tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)

        val_f1, val_hits = test(model, valid_data, cfg['eps'])
        if cfg['lr_schedule']:
            scheduler.step()
        tf_logger.add_scalar('eval_f1', val_f1, epoch)
        tf_logger.add_scalar('eval_hits', val_hits, epoch)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        if val_hits > best_val_hits:
            best_val_hits = val_hits
            torch.save(model.state_dict(), 'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id']))
        print('evaluation best f1:{} current:{}'.format(best_val_f1, val_f1))
        print('evaluation best hits:{} current:{}'.format(best_val_hits, val_hits))

    print('save final model')
    torch.save(model.state_dict(), 'model/{}/{}_final.pt'.format(cfg['name'], cfg['model_id']))

    
    # model_save_path = 'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id'])
    # model.load_state_dict(torch.load(model_save_path))
    
    
    print('..........Finished training, start testing.......')

    test_data = DataLoader(cfg, documents, mode='test')
    model.eval()
    print('finished training, testing final model...')
    test(model, test_data, cfg['eps'])

    print('testing best model...')
    model_save_path = 'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id'])
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test(model, test_data, cfg['eps'])


def test(model, test_data, eps):

    model.eval()
    batcher = test_data.batcher()
    id2entity = test_data.id2entity
    f1s, hits = [], []
    questions = []
    pred_answers = []
    for feed in batcher:
        _, pred, pred_dist = model(feed)
        acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
        batch_size = pred_dist.size(0)
        batch_answers = feed['answers_']
        questions += feed['questions_']
        batch_candidates = feed['candidate_entities']
        pad_ent_id = len(id2entity)
        for batch_id in range(batch_size):
            answers = batch_answers[batch_id]
            candidates = batch_candidates[batch_id,:].tolist()
            probs = pred_dist[batch_id, :].tolist()
            candidate2prob = {}
            for c, p in zip(candidates, probs):
                if c == pad_ent_id:
                    continue
                else:
                    candidate2prob[c] = p
            f1, hit = f1_and_hits(answers, candidate2prob, eps)
            best_ans = get_best_ans(candidate2prob)
            best_ans = id2entity.get(best_ans, '')

            pred_answers.append(best_ans)
            f1s.append(f1)
            hits.append(hit)
    print('evaluation.......')
    print('how many eval samples......', len(f1s))
    print('avg_f1', np.mean(f1s))
    print('avg_hits', np.mean(hits))



    model.train()
    return np.mean(f1s), np.mean(hits)

if __name__ == "__main__":
    # config_file = sys.argv[2]
    cfg = get_config()
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    if cfg['mode'] == 'train':
        train(cfg)
    elif cfg['mode'] == 'test':
        documents = load_documents(cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])
        test_data = DataLoader(cfg, documents, mode='test')
        model = KAReader(cfg)
        model = model.to(torch.device('cuda'))
        model_save_path = 'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id'])
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        test(model, test_data, cfg['eps'])
    else:
        assert False, "--train or --test?"