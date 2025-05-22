import json

def extract_bookmarks_info(file_path):
    """
    从Bookmarks.json文件中提取name和URL
    Args:
        file_path (str): Bookmarks.json文件路径
    Returns:
        list: 包含(name, url)元组的列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bookmarks = []
        
        def traverse(node):
            if 'children' in node:
                for child in node['children']:
                    traverse(child)
            elif node.get('type') == 'url':
                name = node.get('name', '')
                url = node.get('url', '')
                if name and url:
                    bookmarks.append((name, url))
        
        for root in data['roots'].values():
            traverse(root)
            
        return bookmarks
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return []

if __name__ == "__main__":
    bookmarks = extract_bookmarks_info("Bookmarks.json")
    for name, url in bookmarks:
        print(f"名称: {name}\nURL: {url}\n")