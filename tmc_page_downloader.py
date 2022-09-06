import requests
import time

def save_html(html, file_name):
    with open('TMC_pages/tmc_' + str(file_name), 'wb') as f:
        f.write(html)

base_url = 'https://teslamotorsclub.com/tmc/threads/australian-model-3-waiting-room.248573/page-'

last_page_num = 2

for i in range(1,last_page_num+1):
    page_num = i
    url = base_url + str(page_num)
    content = requests.get(url).content

    save_html(content, page_num)

    time.sleep(1)