import json
import jsonlines
import pandas as pd
import os
import time
def pro_train():
    datas = pd.read_csv('trainset0920.csv').values.tolist()
    with jsonlines.open('processed_data_train_new.jsonl','w')as writer:
        for data in datas:
            da = {}
            da['id'] = data[0]
            da['location'] = data[1]
            da['statement'] = data[2]
            da['choices_solve'] = data[3].split('；')
            da['final_solve'] = data[4]
            if da['final_solve']=='龙岗区智慧城市运行中心':
                da['final_solve']='龙岗区智慧城区运行中心'
            if da['final_solve']!=None and da['final_solve']!='龙岗区智慧城区运行中心' and da['final_solve'] in da['choices_solve']:
                writer.write(da)
def pro_test():
    datas = pd.read_csv('testset0920.csv').values.tolist()
    with jsonlines.open('processed_data_test_new.jsonl','w')as writer:
        for data in datas:
            da = {}
            da['id'] = data[0]
            da['location'] = data[1]
            da['statement'] = data[2]
            da['choices_solve'] = data[3].split('；')
            da['final_solve'] = data[4]
            if da['final_solve']=='龙岗区智慧城市运行中心':
                da['final_solve']='龙岗区智慧城区运行中心'
            if da['final_solve']!=None and da['final_solve']!='龙岗区智慧城区运行中心'and  da['final_solve'] in da['choices_solve']:
                writer.write(da)
            else:
                print(da['final_solve'],'    ',da['choices_solve'])
def inform():
    datas = pd.read_json('processed_data_train_new.jsonl',lines = True).values.tolist()
    labels = []
    texts = []
    all = {}
    num=0
    for data in datas:
        labels.append(data[4])
        texts.append(data[2])
        if data[4] not in all.keys():
            all[data[4]] = 1
        if data[4]=='龙岗区智慧城区运行中心':
            num+=1
    print(num)
    # return ((all))
def getTextAndLabel(path):
    datas = pd.read_json(path,lines = True).values.tolist()
    labels = []
    texts = []
    id = []
    all = {}
    choices = []
    for data in datas:
        labels.append(data[4])
        texts.append(data[2])
        choices.append(data[3])
        id.append(data[0])
        if data[4] not in all.keys():
            all[data[4]] = 1
    return labels,texts,all,id,choices
def change_labels(labels,change):
    change_label = []
    for label in labels:
        a = [0 for i in range(54)]
        a[change[label]]=1
        change_label.append(a)
    return change_label
def get_labels_number(all):
    k=0
    labels_number = {}
    for item in all.items():
        labels_number[item[0]] = k
        k+=1
    return labels_number
# pro_train()
# pro_test()
# tt = inform()
# pro_test()
# pro_train()
# labels,texts,all,id,choices=getTextAndLabel('processed_data_train_new.jsonl')
# print(labels)
pro_train()
pro_test()
# labels_number = {'区住房建设局': 0, '市市场监督管理局龙岗监管局': 1, '市交通运输局龙岗管理局': 2, '区人力资源局': 3, '龙岗公安分局': 4, '龙岗区消防救援大队': 5, '区消委会': 6, '市生态环境局龙岗管理局': 7, '区妇联': 8, '区教育局': 9, '区水务局': 10, '区建筑工务署': 11, '龙岗区卫生健康局': 12, '区轨道办': 13, '区城市管理和综合执法局': 14, '市规划和自然资源局龙岗管理局': 15, '区文化广电旅游体育局': 16, '区城市更新和土地整备局': 17, '区规划土地监察局': 18, '龙岗社保分局': 19, '龙岗供电局': 20, '区国资局': 21, '区工业和信息化局': 22, '区委宣传部': 23, '区民政局': 24, '龙岗交警大队': 25, '区委政法委': 26, '区残联': 27, '龙岗邮政分局': 28, '龙岗区税务局': 29, '区投资控股集团有限公司': 30, '区重点区域署': 31, '区委统战部': 32, '区法院': 33, '区科技创新局': 34, '市医疗保障局龙岗分局': 35, '区委组织部': 36, '区政务服务数据管理局': 37, '团区委': 38, '区财政局': 39, '区发展改革局': 40, '区统计局': 41, '电信龙岗分公司': 42, '区应急管理局': 43, '区司法局': 44, '区信访局': 45, '区投资推广和企业服务中心': 46, '区产业投资服务集团有限公司': 47, '区城市建设投资集团有限公司': 48, '区检察院': 49, '区总工会': 50, '区机关事务管理局': 51, '区退役军人事务局': 52, '区委党校': 53,'龙岗区智慧城区运行中心':54}
# labels_number_finpart = {0: '区住房建设局', 1: '市市场监督管理局龙岗监管局', 2: '市交通运输局龙岗管理局', 3: '区人力资源局', 4: '龙岗公安分局', 5: '龙岗区消防救援大队', 6: '区消委会', 7: '市生态环境局龙岗管理局', 8: '区妇联', 9: '区教育局', 10: '区水务局', 11: '区建筑工务署', 12: '龙岗区卫生健康局', 13: '区轨道办', 14: '区城市管理和综合执法局', 15: '市规划和自然资源局龙岗管理局', 16: '区文化广电旅游体育局', 17: '区城市更新和土地整备局', 18: '区规划土地监察局', 19: '龙岗社保分局', 20: '龙岗供电局', 21: '区国资局', 22: '区工业和信息化局', 23: '区委宣传部', 24: '区民政局', 25: '龙岗交警大队', 26: '区委政法委', 27: ' 区残联', 28: '龙岗邮政分局', 29: '龙岗区税务局', 30: '区投资控股集团有限公司', 31: '区重点区域署', 32: '区委统战部', 33: '区法院', 34: '区科技创新局', 35: '市医疗保障局龙岗分局', 36: '区委组织部', 37: '区政务服务数据管理局', 38: '团区委', 39: '区财政局', 40: '区发展改革局', 41: '区统计局', 42: '电信龙岗分公司', 43: '区应急管理局', 44: '区司法局', 45: '区信访局', 46: '区投资推广和企业服务中心', 47: '区产业投资服务集团有限公司', 48: '区城市建设投资集团有限公司', 49: '区检察院', 50: '区总工会', 51: '区机关事务管理局', 52: '区退役军人事务局', 53: '区委党校',54:'龙岗区智慧城区运行中心'}
labels_number = {'区住房建设局': 0, '市市场监督管理局龙岗监管局': 1, '市交通运输局龙岗管理局': 2, '区人力资源局': 3, '龙岗公安分局': 4, '龙岗区消防救援大队': 5, '区消委会': 6, '市生态环境局龙岗管理局': 7, '区妇联': 8, '区教育局': 9, '区水务局': 10, '区建筑工务署': 11, '龙岗区卫生健康局': 12, '区轨道办': 13, '区城市管理和综合执法局': 14, '市规划和自然资源局龙岗管理局': 15, '区文化广电旅游体育局': 16, '区城市更新和土地整备局': 17, '区规划土地监察局': 18, '龙岗社保分局': 19, '龙岗供电局': 20, '区国资局': 21, '区工业和信息化局': 22, '区委宣传部': 23, '区民政局': 24, '龙岗交警大队': 25, '区委政法委': 26, '区残联': 27, '龙岗邮政分局': 28, '龙岗区税务局': 29, '区投资控股集团有限公司': 30, '区重点区域署': 31, '区委统战部': 32, '区法院': 33, '区科技创新局': 34, '市医疗保障局龙岗分局': 35, '区委组织部': 36, '区政务服务数据管理局': 37, '团区委': 38, '区财政局': 39, '区发展改革局': 40, '区统计局': 41, '电信龙岗分公司': 42, '区应急管理局': 43, '区司法局': 44, '区信访局': 45, '区投资推广和企业服务中心': 46, '区产业投资服务集团有限公司': 47, '区城市建设投资集团有限公司': 48, '区检察院': 49, '区总工会': 50, '区机关事务管理局': 51, '区退役军人事务局': 52, '区委党校': 53}
labels_number_finpart = {0: '区住房建设局', 1: '市市场监督管理局龙岗监管局', 2: '市交通运输局龙岗管理局', 3: '区人力资源局', 4: '龙岗公安分局', 5: '龙岗区消防救援大队', 6: '区消委会', 7: '市生态环境局龙岗管理局', 8: '区妇联', 9: '区教育局', 10: '区水务局', 11: '区建筑工务署', 12: '龙岗区卫生健康局', 13: '区轨道办', 14: '区城市管理和综合执法局', 15: '市规划和自然资源局龙岗管理局', 16: '区文化广电旅游体育局', 17: '区城市更新和土地整备局', 18: '区规划土地监察局', 19: '龙岗社保分局', 20: '龙岗供电局', 21: '区国资局', 22: '区工业和信息化局', 23: '区委宣传部', 24: '区民政局', 25: '龙岗交警大队', 26: '区委政法委', 27: ' 区残联', 28: '龙岗邮政分局', 29: '龙岗区税务局', 30: '区投资控股集团有限公司', 31: '区重点区域署', 32: '区委统战部', 33: '区法院', 34: '区科技创新局', 35: '市医疗保障局龙岗分局', 36: '区委组织部', 37: '区政务服务数据管理局', 38: '团区委', 39: '区财政局', 40: '区发展改革局', 41: '区统计局', 42: '电信龙岗分公司', 43: '区应急管理局', 44: '区司法局', 45: '区信访局', 46: '区投资推广和企业服务中心', 47: '区产业投资服务集团有限公司', 48: '区城市建设投资集团有限公司', 49: '区检察院', 50: '区总工会', 51: '区机关事务管理局', 52: '区退役军人事务局',53: '区委党校'}