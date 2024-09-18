import requests
from bs4 import BeautifulSoup as bs
import os
from tqdm import tqdm 
import unicodedata
import json
from urllib.parse import urljoin
from haystack.nodes import TransformersTranslator
from utils import split_and_translate, translate_docs, get_true_case
from typing import List

class Crawler:

    def __init__(self, urls:List[str] = [], keywords:List[str] = []):
        self.visited_urls = []
        self.new_urls = urls
        self.keywords = keywords

    def get_linked_urls(self, url, response):
        soup = bs(response.content, 'lxml')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path.startswith('/'): path = urljoin(url, path)     
            if self.keywords:
                if all(substring in path for substring in self.keywords):
                    yield path
                else:
                    continue
            yield path

    def add_new_url(self, url:str):
        if url not in self.visited_urls and url not in self.new_urls:
            self.new_urls.append(url)

    def crawl(self):
        # get a new url from the list, find all linked urls in url and update the list, discard url 
        url = self.new_urls.pop(0)
        try:
            response = requests.get(url)
            linked_urls = self.get_linked_urls(url, response)
            for linked_url in linked_urls:
                self.add_new_url(linked_url)
        except Exception:
            pass
        self.visited_urls.append(url)
    
    def get_crawled_urls(self):
        # crawl the initial given url, crawl all the linked urls found in the initial url and then stop crawling
        stop = 0
        while len (self.visited_urls) == 0:
            print ("Crawling initial url ... ")  
            self.crawl()
            stop += len (self.new_urls) 
        else:
            print ("Crawling linked urls ... ")
            while stop != 0:
                self.crawl()
                stop -= 1
        if self.new_urls:
            self.visited_urls = list (set(self.visited_urls + self.new_urls))
        return self.visited_urls

    @staticmethod
    def extract_NPHO_data(urls:List[str], save_dir:str = 'data/npho/'):
        # extract plain text from eody web page
        output_file = save_dir + 'docs.txt'
        open (output_file, 'w', encoding='utf8').close()

        for url in tqdm (urls, desc= 'Extracting Data ...'): 
            
            response = requests.get(url)
            if response.headers.get('content-type') == 'text/html; charset=UTF-8':
                soup = bs(response.content, "lxml")  
                try:
                    
                    p_tags  = soup.find ("main", {'id': 'main-container'}).find_all('p')
                    for p in p_tags:
                        text =    ' '.join (p.find_all(text=True))
                        text = unicodedata.normalize ("NFKD", text)     
                        text = get_true_case(text)
                        with open(output_file,'a') as f:                 
                            f.write(text)
                            f.write('\n')
                except Exception:
                    continue
            else: continue

    @staticmethod
    def extract_WHO_data (urls:List[str], translate:bool = True, save_dir:str = 'data/who/'):
    # extract qa pairs from who webpage

        # validate urls
        criterion = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub"
        
        for url in urls:
            if criterion not in url:
                raise Exception(f"All given urls must contain {criterion}")
        
        # create save files
        if translate:
            suffix = 'el'
        else:
            suffix = 'en'
        output_files = [save_dir + f'queries1.{suffix}.txt', save_dir + f'answers1.{suffix}.txt']
        for f in output_files: 
            open(f, 'w').close()

        for url in tqdm(urls, desc='Extracting Data ...'):

            queries = []
            answers = []
            
            response = requests.get(url)
            soup = bs (response.content, "lxml")
            faqs = soup.find_all('div', {"class": "sf-accordion__panel"})        

            for faq in faqs:
                query =  faq.find('div', {"class": "sf-accordion__trigger-panel"}).get_text().replace('\n', "").replace('\n', " ")
                answer = faq.find ('div', {"class":"sf-accordion__content"}).get_text().replace('\n', "").replace('\n', " ")
                answers.append(unicodedata.normalize ("NFKD", answer))
                queries.append(unicodedata.normalize ("NFKD", query))
            
            if queries and answers:

                if translate:
                    queries  = translate_docs(queries)
                    answers = split_and_translate(answers)
            
                with open (output_files[0], 'a', encoding='utf-8') as f:
                    for q in queries:
                        f.write(q)
                        f.write('\n')
                with open (output_files[1], 'a', encoding='utf-8') as f:
                    for a in answers:
                        f.write(a)
                        f.write('\n')
            else: 
                continue

if __name__ == '__main__': 
    WHO_URL = 'https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub'

    #crawler = Crawler(urls=[EODY_URL], keywords=['eody.gov.gr', 'covid-19'])
    #eody_urls = crawler.get_crawled_urls()
    #print (eody_urls)
    #crawler = Crawler (urls=[WHO_URL], keywords=[WHO_URL])
    #who_urls = crawler.get_crawled_urls()
    eody_urls = ['https://eody.gov.gr/covid-19-odigies-gia-mesa-mazikis-metaforas/', 'https://eody.gov.gr/odigies-tis-ethnikis-epitropis-emvoliasmon-gia-ti-chorigisi-emvolioy-covid-19-se-atoma-poy-lamvanoyn-farmakeytiki-agogi/', 'https://eody.gov.gr/covid-19-faq-20220106/', 'https://eody.gov.gr/covid-19-eppag-ygeias-ektimisi-kindinou/', 'https://eody.gov.gr/evdomadiaia-ekthesi-epitirisis-covid-19-21-noemvrioy-2022-27-noemvrioy-2022/', 'https://eody.gov.gr/odigies-tis-ethnikis-epitropis-emvoliasmon-gia-ti-chorigisi-emvolioy-covid-19-se-atoma-poy-lamvanoyn-farmakeytiki-agogi/?print=print', 'https://eody.gov.gr/covid-19-vrefonip_stath-20220524/', 'https://eody.gov.gr/en/novel-coronavirus-covid-19-questions-and-answers-for-the-public/', 'https://eody.gov.gr/en/covid-19-seaport-personnel/', 'https://eody.gov.gr/category/covid-19/?yeararchive=2023', 'https://eody.gov.gr/covid-19-odigies-gia-to-epivatiko/?print=print', 'https://eody.gov.gr/covid-19-krouazieroploia-202206/', 'https://eody.gov.gr/covid-19-egkymosini-emvoliasmos-20220628/', 'https://eody.gov.gr/odigies-apomonosis-karantinas-kroysmaton-covid-19-kai-epafon-toys/', 'https://eody.gov.gr/en/covid-19/', 'https://eody.gov.gr/covid-19-odigies_gia_prosopiko_edafoys_aerodromion-202212/', 'https://eody.gov.gr/covid-19-prosopiko-limenon-202206/', 'https://eody.gov.gr/en/covid-19-advice-for-airport-ground-personnel/', 'https://eody.gov.gr/category/covid-19/page/3/', 'https://eody.gov.gr/en/further-covid-19-data-from-greece/', 'https://eody.gov.gr/category/covid-19/?yeararchive=2020', 'https://eody.gov.gr/kathorismos-omadon-ayximenoy-kindynoy-gia-sovari-loimoxi-covid-19', 'https://eody.gov.gr/category/covid-19/?yeararchive=2022', 'https://eody.gov.gr/kathorismos-omadon-ayximenoy-kindynoy-gia-sovari-loimoxi-covid-19/?print=print', 'https://eody.gov.gr/covid-19-kataskinosi-genikes-20220627/', 'https://eody.gov.gr/covid-19-odigies-hdika/', 'https://eody.gov.gr/en/covid-19/?print=print', 'https://eody.gov.gr/koronoios-covid-19-odigies-psychologikis-ypostirixis-ton-politon/', 'https://eody.gov.gr/en/covid-19-advice-for-hotels-in-quarantine/', 'https://eody.gov.gr/covid-19-odigies-gia-mesa-mazikis-metaforas/?print=print', 'https://eody.gov.gr/koronoios-covid-19-odigies-psychologikis-ypostirixis-ton-politon/?print=print', 'https://eody.gov.gr/covid-19-maska/', 'https://eody.gov.gr/category/covid-19/page/2/', 'https://eody.gov.gr/covid-19-faq-20220106/?print=print', 'https://eody.gov.gr/neos-koronaios-covid-19-odigies-gia-taxidiotes/', 'https://eody.gov.gr/loimoxi-apo-to-neo-koronoio-covid-19-odigies-profylaxis-gia-to-koino/', 'https://eody.gov.gr/neos-koronaios-covid-19/?print=print', 'https://eody.gov.gr/category/covid-19/', 'https://eody.gov.gr/covid-19-odigies-mfi/', 'https://eody.gov.gr/covid-19-ergastiriaki-diagnosi/', 'https://eody.gov.gr/covid-19-algorithmos-egkyes/', 'https://eody.gov.gr/covid-19-odigies-gia-to-epivatiko-koino-se-mesa-mazikis-metaforas-metro-leoforeia-trolei-proastiakos-astikos-sidirodromos/', 'https://eody.gov.gr/covid-19-odigies_leoforeia-20220607/', 'https://eody.gov.gr/en/infection-from-new-coronavirus-sars-cov-2-covid-19-advice-to-private-physicians/', 'https://eody.gov.gr/en/category/covid-19-en/', 'https://eody.gov.gr/covid-19-thylasmos-loimoxi-20220628/', 'https://eody.gov.gr/covid-19-kataskinosi-diaxeirisi-20220627/', 'https://eody.gov.gr/category/covid-19/?yeararchive=2021', 'https://eody.gov.gr/en/novel-coronavirus-covid-19-advice-for-travellers/', 'https://eody.gov.gr/odigies-apomonosis-karantinas-kroysmaton-covid-19-kai-epafon-toys/?print=print', 'https://eody.gov.gr/covid-19-odigies-gia-ti-chrisi-maskas-apo-to-koino/?print=print', 'https://eody.gov.gr/covid-19-odigies-mxa-mtn/', 'https://eody.gov.gr/evdomadiaia-ekthesi-epitirisis-covid-19-10-oktovrioy-2022-16-oktovrioy-2022/', 'https://eody.gov.gr/covid-19-odigies-eel-paxlovid-20221017/', 'https://eody.gov.gr/covid-19-nekra-somata-202102/', 'https://eody.gov.gr/odigies-tis-ethnikis-epitropis-emvoliasmon-gia-ti-chorigisi-emvolioy-covid-19-se-atoma-poy-anaferoyn-istoriko-sovaris-anafylaxias/', 'https://eody.gov.gr/en/current-state-of-covid-19-outbreak-in-greece-and-timeline-of-key-containment-events/', 'https://eody.gov.gr/kathorismos-omadon-ayximenoy-kindynoy-gia-sovari-loimoxi-covid-19/', 'https://eody.gov.gr/covid-19-odigies-gia-ti-chrisi-maskas-apo-to-koino/', 'https://eody.gov.gr/odigies-tis-ethnikis-epitropis-emvoliasmon-gia-ti-chorigisi-emvolioy-covid-19-se-atoma-poy-anaferoyn-istoriko-sovaris-anafylaxias/?print=print', 'https://eody.gov.gr/neos-koronaios-covid-19/', 'https://eody.gov.gr/category/covid-19/page/80/', 'https://eody.gov.gr/covid-19-odigies-gia-to-epivatiko/', 'https://eody.gov.gr/loimoxi-apo-to-neo-koronoio-covid-19-odigies-profylaxis-gia-to-koino/?print=print', 'https://eody.gov.gr/covid-19-traina-20220607/', 'https://eody.gov.gr/odigies-tis-ethnikis-epitropis-emvoliasmon-gia-ti-chorigisi-emvolioy-covid-19-kata-tin-kyisi-kai-ton-thilasmo/?print=print', 'https://eody.gov.gr/wp-content/uploads/2021/02/orismoi-kroysmatos-covid-19-kai-epafon-kroysmatos-covid-19-202102.pdf', 'https://eody.gov.gr/covid-19-odigies-omadika-athlimata-20220301/', 'https://eody.gov.gr/evdomadiaia-ekthesi-epitirisis-covid-19-12-dekemvrioy-2022-18-dekemvrioy-2022/', 'https://eody.gov.gr/covid-19-odigies-krousma-ploio-20220602/', 'https://eody.gov.gr/covid-19-egkymosini-loimoxi-20220628/', 'https://eody.gov.gr/odigies-tis-ethnikis-epitropis-emvoliasmon-gia-ti-chorigisi-emvolioy-covid-19-kata-tin-kyisi-kai-ton-thilasmo/', 'https://eody.gov.gr/covid-19-ipertasi/', 'https://eody.gov.gr/neos-koronaios-covid-19-odigies-gia-taxidiotes/?print=print', 'https://eody.gov.gr/covid-19-aeroygeionomeia_pliromata_202212/', 'https://eody.gov.gr/en/infection-from-new-coronavirus-sars-cov-2-covid-19-guidance-for-healthcare-settings/']
    