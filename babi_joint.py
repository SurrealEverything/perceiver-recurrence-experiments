from glob import glob
from os import path
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import re
import numpy as np
from sklearn.utils import shuffle


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def pad_collate(batch):
    max_context_len = 70 * 13 + 13
    # max_question_len = 13
    for i, elem in enumerate(batch):
        _context, question, answer, task = elem
        _context = [wrd for sent in _context for wrd in sent]
        _context.extend([wrd for wrd in question])
        _context = _context[-max_context_len:]
        context = np.pad(_context, (0, max_context_len - len(_context)), 'constant', constant_values=0)
        # question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, answer, task)
    return default_collate(batch)


class BabiDataset(Dataset):
    def __init__(self, mode='train', ds_path='data/en-10k/qa{}_*', vocab_path='dataset/babi_vocab.pkl'):
        self.vocab_path = vocab_path
        self.mode = mode
        raw_train, raw_test = get_raw_babi(ds_path=ds_path)
        self.QA = adict()
        self.QA.VOCAB = {'<PAD>': 0, '<EOS>': 1}
        self.QA.IVOCAB = {0: '<PAD>', 1: '<EOS>'}
        self.train = self.get_indexed_qa(raw_train)
        self.valid = [self.train[i][int(-len(self.train[i])/10):] for i in range(4)]
        self.train = [self.train[i][:int(9 * len(self.train[i])/10)] for i in range(4)]
        self.test = self.get_indexed_qa(raw_test)

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        elif self.mode == 'valid':
            return len(self.valid[0])
        elif self.mode == 'test':
            return len(self.test[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers, tasks = self.train
        elif self.mode == 'valid':
            contexts, questions, answers, tasks = self.valid
        elif self.mode == 'test':
            contexts, questions, answers, tasks = self.test
        return contexts[index], questions[index], answers[index], tasks[index]

    def get_indexed_qa(self, raw_babi):
        unindexed = get_unindexed_qa(raw_babi)
        questions = []
        contexts = []
        answers = []
        tasks = []
        for qa in unindexed:
            context = [c.lower().split() + ['<EOS>'] for c in qa['C']]

            for con in context:
                for token in con:
                    self.build_vocab(token)
            context = [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
            question = qa['Q'].lower().split() + ['<EOS>']
            
            for token in question:
                self.build_vocab(token)

            question = [self.QA.VOCAB[token] for token in question]

            self.build_vocab(qa['A'].lower())
            answer = self.QA.VOCAB[qa['A'].lower()]

            contexts.append(context)
            questions.append(question)
            answers.append(answer)
            tasks.append(qa['T'])
        return (shuffle(contexts, questions, answers, tasks))

    def build_vocab(self, token):
        if not token in self.QA.VOCAB:
            next_index = len(self.QA.VOCAB)
            self.QA.VOCAB[token] = next_index
            self.QA.IVOCAB[next_index] = token
   
def get_raw_babi(ds_path):
    train=''
    test=''
    for task_id in range(1, 21):
        paths = glob(ds_path.format(task_id))
        for path in paths:
            if 'train' in path:
                train += 'T' + str(task_id) + '\n'
                with open(path, 'r') as fp:
                    train_task = fp.read() + '\n'
                    train+=train_task                    
            elif 'test' in path:
                test += 'T' + str(task_id) + '\n'
                with open(path, 'r') as fp:
                    test_task = fp.read() + '\n'
                    test+=test_task
                    
    return train, test
            
def get_unindexed_qa(raw_babi):
    tasks = []
    task = None
    babi = raw_babi.strip().split('\n')
    for i, line in enumerate(babi):
        if len(line) == 0:
            continue
        if line[0] == 'T':
            task_id = int(line[1:])
            continue
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": "", 'T': ""}
            counter = 0
            id_map = {}

        line = line.strip()
        line = line.replace('.', '')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line + '<line>'
            id_map[id] = counter
            counter += 1
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = [] # Supporting facts
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tc = task.copy()
            tc['C'] = tc['C'].split('<line>')[:-1]
            tc['T'] = task_id
            tasks.append(tc)
    return tasks

# babi_train = BabiDataset(ds_path='/home/gabriel/Documents/datasets/bAbi/en/qa{}_*', 
#                          vocab_path='/home/gabriel/Documents/datasets/bAbi/en/babi{}_vocab.pkl')
