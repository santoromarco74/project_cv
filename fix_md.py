import re

with open('c:\\Users\\saintgold\\histreg\\relazione\\relazione_sorgente.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix section references
content = content.replace('(§9.1)', '(Sezione 9.1)')
content = content.replace('(§4)', '(Sezione 4)')
content = content.replace('(§3)', '(Sezione 3)')
content = content.replace('§9.1', 'Sezione 9.1')
content = content.replace('§4', 'Sezione 4')
content = content.replace('§3', 'Sezione 3')
content = content.replace('§5', 'Sezione 5')
content = content.replace('§12.2', 'Sezione 12.2')
content = content.replace('§7.1', 'Sezione 7.1')
content = content.replace('§6.1', 'Sezione 6.1')

# Fix bold math
content = content.replace(r'$\mathbf{1.1 \cdot 10^{-13}} px$', r'$\mathbf{1.1 \times 10^{-13}}\ \text{px}$')
content = content.replace(r'$\mathbf{10^{-9}} px$', r'$\mathbf{10^{-9}}\ \text{px}$')

with open('c:\\Users\\saintgold\\histreg\\relazione\\relazione_sorgente.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')