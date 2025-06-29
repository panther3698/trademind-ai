import os, ast, re
from pathlib import Path
from collections import defaultdict
import colorama
colorama.init()

def main():
    app_dir = Path('app')
    all_py = list(app_dir.rglob('*.py'))
    imports = defaultdict(set)
    used = set()
    orphaned = []
    file_info = {}
    for f in all_py:
        src = f.read_text(encoding='utf-8', errors='ignore')
        try:
            tree = ast.parse(src, filename=str(f))
        except Exception:
            continue
        imps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imps.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imps.add(node.module.split('.')[0])
        file_info[str(f)] = {
            'lines': src.count('\n')+1,
            'size': os.path.getsize(f),
            'imports': imps,
            'has_code': any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.iter_child_nodes(tree)),
            'commented': bool(re.search(r'^\s*#.*$', src, re.M))
        }
        imports[str(f)] = imps
        used |= imps
    orphaned = [f for f in all_py if not any(f.stem in v for v in imports.values() if str(f) != v) and file_info[str(f)]['has_code']]
    safe = [f for f in orphaned if not file_info[str(f)]['has_code']]
    review = [f for f in orphaned if file_info[str(f)]['has_code']]
    keep = [f for f in all_py if f.name in ['main.py','service_manager.py'] or 'api/routes' in str(f) or 'core/services' in str(f)]
    def fmt(f, c):
        return f"{c}{str(f.relative_to(app_dir.parent))} ({file_info[str(f)]['lines']} lines, {file_info[str(f)]['size']/1024:.1f} KB){colorama.Style.RESET_ALL}"
    print(f'\n\033[1mTRADEMIND AI CODEBASE CLEANUP ANALYSIS\033[0m')
    print(f"\033[92mSAFE TO REMOVE ({len(safe)} files, {sum(file_info[str(f)]['size'] for f in safe)/1024:.1f} KB):\033[0m")
    [print(fmt(f, '\033[92m')) for f in safe]
    print(f"\n\033[93mREVIEW NEEDED ({len(review)} files, {sum(file_info[str(f)]['size'] for f in review)/1024:.1f} KB):\033[0m")
    [print(fmt(f, '\033[93m')) for f in review]
    print(f"\n\033[91mKEEP - CRITICAL FILES:\033[0m")
    [print(fmt(f, '\033[91m')) for f in keep]
    print(f"\nCLEANUP IMPACT:\nTotal files analyzed: {len(all_py)}\nSafe to remove: {len(safe)} files ({sum(file_info[str(f)]['size'] for f in safe)/1024:.1f} KB)\nPotential space saved: {sum(file_info[str(f)]['size'] for f in safe+review)/1024:.1f} KB\nEstimated cleanup benefit: {'High' if len(safe)+len(review)>10 else 'Medium' if len(safe)+len(review)>3 else 'Low'}\n\nRECOMMENDED ACTIONS:\n- Remove SAFE files immediately\n- Review files marked for REVIEW manually\n- Clean unused imports within files\n")

if __name__ == "__main__":
    main() 