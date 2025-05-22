const fs = require('fs');

function extractBookmarksInfo(filePath) {
    try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        const bookmarks = [];
        
        function traverse(node) {
            if (node.children) {
                node.children.forEach(child => traverse(child));
            } else if (node.type === 'url') {
                const name = node.name || '';
                const url = node.url || '';
                if (name && url) {
                    bookmarks.push({name, url});
                }
            }
        }
        
        Object.values(data.roots).forEach(root => traverse(root));
        return bookmarks;
    } catch (e) {
        console.error(`处理文件时出错: ${e}`);
        return [];
    }
}

const bookmarks = extractBookmarksInfo('Bookmarks.json');
bookmarks.forEach(({name, url}) => {
    console.log(`名称: ${name}\nURL: ${url}\n`);
});