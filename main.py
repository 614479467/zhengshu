import json
import jsonlines
import torch
from train import CustomerDataset_train,Model_train,CustomerDataset_test
from processes_data import *
from transformers import RobertaTokenizer, RobertaModel
import argparse
from torch.utils.data import DataLoader, Dataset
import time
def train(lr):
    #max = 0.857142
    tt = 277
    tt1 = 208
    max = 0.8626
    labels,train_texts,all,id,choices= getTextAndLabel('processed_data_train_new.jsonl')
    train_labels = change_labels(labels,labels_number)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_dataset = CustomerDataset_train(train_texts, train_labels, tokenizer,id= id,max_length=128,choices=choices)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last= True)
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model_train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(epoch)
        model.train()
        total_loss = 0
        step = 0
        all_loss = []
        time1 = time.time()
        for batch in train_dataloader: 
            step+=1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = torch.tensor(batch['label']).to(float).to(device)
            choices_nhot = batch['choices_nhot'].to(device)
            optimizer.zero_grad()
            outputs = model.forward(input_ids, attention_mask,choices_nhot)
            loss = criterion(outputs,labels)
            total_loss += loss.item()
            print('Step = ',step,":::::",'loss = ',loss.item())
            if step%64==0:
                print('Step = ',step,":::::",'loss = ',loss.item())
                all_loss.append(loss.item())
                with open('loss_file.txt','a')as f:
                    a = str([epoch,step,loss.item()])[1:-2]+'\n'
                    f.write(a)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_dataloader)}")
        time2 = time.time()
        print(time2-time1)
        dir = os.listdir('./')
        torch.save(model.state_dict(),'./model_add_choices.pth')
        eval()
        acc = getacc()
        print(acc)
        print(max)
        if acc>max:
            torch.save(model.state_dict(),'./model_add_choices_best.pth')
            max = acc
    print(max)



def eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model_train().to(device)
    model.load_state_dict(torch.load('./model_add_choices.pth'))
    model.eval()
    labels,test_texts,all,id,choices= getTextAndLabel('processed_data_test_new.jsonl')
    test_labels = change_labels(labels,labels_number)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_dataset = CustomerDataset_test(test_texts, test_labels, tokenizer,id= id,choices = choices, max_length=128,)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    with open('result_add.txt','w',encoding = 'utf-8')as f:
        for batch in test_dataloader: 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = torch.tensor(batch['label']).to(float).to(device)
            choices_nhot = batch['choices_nhot'].to(float).to(device)
            output = model.forward(input_ids, attention_mask,choices_nhot)
            output_list = output.tolist()
            choice = [da[0] for da in batch['choices']]
            output_id = torch.argmax(output).item()
            flag = True
            old_id=-1
            while labels_number_finpart[output_id] not in choice and flag:
                output_list[0][output_id]=0
                output = torch.tensor(output_list)
                output_id = torch.argmax(output).item()
                if old_id==output_id:
                    flag = False
                old_id = output_id
            f.write(str(batch['id'])[2:-2]+' '+str(labels_number_finpart[output_id])+'\n')
def getacc():
    data = pd.read_table('result_add.txt',' ').values.tolist()
    datas = pd.read_json('processed_data_test_new.jsonl',lines=True).values.tolist()
    dic = {}
    acc_num = 0
    for da in datas:
        dic[str(da[0])]=da[4]
    for da in data:
        if (da[1]==dic[da[0]]):
            acc_num+=1
    return (acc_num/len(data))
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--mode',
    #     required = True,
    #     type = str,
    #     help = "train or test",
    # )
    # parser.add_argument(
    #     '--lr',
    #     type = float,
    #     default = 0.01,
    #     help = "learning rate",
    # )
    # args = parser.parse_args()
    # if args.mode!='train' and args.mode!='eval':
    #     raise 'You have to choose train or eval'
    # else:
    #     if args.mode=='train':
    #         train(args.lr)
    #     else:
    #         getacc()
    # eval()
    # print(getacc())
    train(0.0001)