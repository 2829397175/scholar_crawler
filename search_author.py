from serpapi import GoogleSearch
import requests
import pandas as pd
from pydantic import BaseModel
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import json
import os
import re

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
def calculate_sim(sentence1,sentence2):
    if sentence1.lower() in sentence2.lower():
        return True
    
    # 对句子进行tokenize和padding
    inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)

    # 获取句子的嵌入表示
    with torch.no_grad():
        output1 = model(**inputs1)
        output2 = model(**inputs2)

    # 提取嵌入表示
    embedding1 = output1.last_hidden_state.mean(dim=1).numpy()
    embedding2 = output2.last_hidden_state.mean(dim=1).numpy()

    # 计算余弦相似度
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity>0.9


def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
        json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)
        
def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

class Scholar_Crawler(BaseModel):
    
    article_num = 200
    article_queue = [] # 保存开始爬的queue
    df:Any = None
    
    crawld_articles = {} #id: result
    crawld_authors = {} #id: result
    article_title_map = {} # title:id
    author_name_map = {} # name:id
    
    
    def __init__(self,*args,**kwargs):
        df = pd.read_csv("llm_agent/raw_data.csv")
        crawld_articles = readinfo("crawld/article.json")
        crawld_authors = readinfo("crawld/author.json")
        article_title_map = {} # title:id
        author_name_map = {} # name:id
        
        super().__init__(
            article_queue = df["Title"].to_list(),
            df = df,
            crawld_authors = crawld_authors,
            crawld_articles = crawld_articles,
            article_title_map = article_title_map,
            author_name_map = author_name_map,
            *args,**kwargs
        )
        
    def run(self):
        num_article = 0
        while(num_article<self.article_num and \
            len(self.article_queue)>0):
            title = self.article_queue.pop()
            article_id = self.article_title_map.get(title)
            if article_id in self.crawld_articles.keys():continue
            self.get_article(title)
            num_article += 1
            if(num_article%10==0):
                self.save()
                
        self.save()
                
    def save(self):
        writeinfo(self.crawld_articles,"crawld/article.json")
        writeinfo(self.crawld_authors,"crawld/author.json")                    
        
    def get_author(self,author_id):
        params = {
        "engine": "google_scholar_author",
        "author_id": author_id,
        "api_key": "f5391e0d8460c4216de952f0eaeb07bc12575fdc1ba88e443ec40b7e7f489f99"
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    
    def get_article_neighbour(self,url):
        regex = r"cites=(.*)&engine=google_scholar&hl=en"
        params = {
            "cites": re.search(regex,url).group(1),
            "engine":"google_scholar",
            "api_key": "f5391e0d8460c4216de952f0eaeb07bc12575fdc1ba88e443ec40b7e7f489f99"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results
        
    def get_article(self,article_title):
        
        params = {
        "engine": "google_scholar",
        "q": article_title,
        "api_key": "f5391e0d8460c4216de952f0eaeb07bc12575fdc1ba88e443ec40b7e7f489f99"
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results["organic_results"]
        
        article_info = {}
        for result in organic_results:
            if calculate_sim(result["title"],article_title):
                article_info = result
                break
            
        if article_info != {}:
            publication_infos = article_info["publication_info"]
            self.article_title_map[article_title] = article_info["result_id"]
            self.article_title_map[article_info["title"]] = article_info["result_id"]
            
            
            """crawl authors"""
            for author in publication_infos.get("authors",[]):
                author_name = author["name"]
                author_id = self.author_name_map.get(author_name)
                if author_id in self.crawld_authors.keys():
                    continue
                
                author_info = self.get_author(author_id=author_id)
                self.author_name_map[author_name] = author_info["search_metadata"]["id"]
                self.crawld_authors[author_info["search_metadata"]["id"]] = author_info
                     
            """crawl citations"""
            citation_url = article_info["inline_links"]["cited_by"]["serpapi_scholar_link"]
            # results = self.get_serpurl_results(citation_url) # 对于查找的一阶邻居 如果不在crawld的
            results = self.get_article_neighbour(citation_url)
            article_info["cited_by_articles_info"] = results

            self.crawld_articles[article_info["result_id"]] = article_info
            
            
    def get_serpurl_results(self,serpapi_scholar_link):
        params = {
        "api_key": "f5391e0d8460c4216de952f0eaeb07bc12575fdc1ba88e443ec40b7e7f489f99"
        }
        
        results = requests.get(serpapi_scholar_link,params=params)    
        return results


    

    
    
if __name__ == "__main__":
    
    crawler = Scholar_Crawler()
    crawler.run()
    