import torch
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset
from processes_data import *
class CustomerDataset_train(Dataset):
    def __init__(self, texts, labels, tokenizer,id,max_length,choices):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id = id
        self.choices = choices
 
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).replace('\n',';')
        label = self.labels[idx]
        id = self.id[idx]
        choices = self.choices[idx]
        choices_nhot = [0 for i in range(54)]
        for choice in choices:
            if choice in labels_number.keys():
                choices_nhot[labels_number[choice]]=1
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label),
            'text': text,
            'id':id,
            'choices_nhot':torch.tensor(choices_nhot)
        }
    
class CustomerDataset_test(Dataset):
    def __init__(self, texts, labels, tokenizer,id, max_length,choices):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id = id
        self.choices = choices

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).replace('\n',';')
        label = self.labels[idx]
        id = self.id[idx]
        choices = self.choices[idx]
        choices_nhot = [0 for i in range(54)]
        for choice in choices:
            if choice in labels_number.keys():
                choices_nhot[labels_number[choice]]=1
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label),
            'text': text,
            'id':id,
            'choices':choices,
            'choices_nhot':torch.tensor(choices_nhot) 
        }
class Model_train(torch.nn.Module):
    def __init__(self):
        super(Model_train, self).__init__()
        self.model= RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(self.model.config.hidden_size, 256)
        self.bn = torch.nn.BatchNorm1d(768)
        self.fc1 = (torch.nn.Linear(256,128))
        self.fc2 = (torch.nn.Linear(128,54))
        self.fc3 = (torch.nn.Linear(54,54))
        self.leakrelu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax()
    def forward(self,input_ids,mask,choices):
        outputs =  self.bn(self.model(input_ids = input_ids,attention_mask = mask)['pooler_output'])
        self.input = self.dropout(self.leakrelu(self.fc(outputs)))
        self.input = self.dropout((self.leakrelu(self.fc1(self.input))))
        self.input= self.dropout(self.leakrelu(self.fc2(self.input)))
        self.input +=choices
        self.output= self.dropout(self.softmax(self.fc3(self.input)))
        return self.output

