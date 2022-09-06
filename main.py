from bs4 import BeautifulSoup
import requests

# with open('TMC.html', 'r', encoding='utf8') as html_file:
#     content = html_file.read()

# TODO: Chuck the link iterator into a try catch incase a broken link appears

base_url = 'https://teslamotorsclub.com/tmc/threads/australian-model-3-waiting-room.248573/page-'
page_num = 798
url = base_url + str(page_num)
content = requests.get(url).content

soup = BeautifulSoup(content, 'lxml')

cards =  soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['bbWrapper'])

posts = []

for text in cards:
    post = ''.join(list(text.find_all(text=True, recursive = False))).replace('\n'," ")
    posts.append(post)

print(posts)
