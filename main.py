from bs4 import BeautifulSoup

with open('TMC.html', 'r', encoding='utf8') as html_file:
    content = html_file.read()
    
    soup = BeautifulSoup(content, 'lxml')

    cards =  soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['bbWrapper'])

    posts = []

    for text in cards:
        post = ''.join(list(text.find_all(text=True, recursive = False))).replace('\n'," ")
        posts.append(post)
