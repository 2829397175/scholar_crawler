import requests
search_url = 'https://www.researchgate.net/search/publication?q=Managing+the+Operator+Ordering+Problem+in+Parallel+Databases+This+paper+focuses+on'

search_url = 'https://arxiv.org/abs/2308.04030'

search_url = 'https://scholar.google.com/citations?user=X8gAsg8AAAAJ&hl=zh-CN&oi=sra'
headers = {
            'User-Agent': "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        }



response = requests.get(search_url, headers=headers,verify=False)
response