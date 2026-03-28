import urllib.request
import urllib.parse
import json
import time

def search_ss(query, limit=5):
    q = urllib.parse.quote(query)
    url = (
        f'https://api.semanticscholar.org/graph/v1/paper/search'
        f'?query={q}&limit={limit}'
        f'&fields=title,authors,year,venue,externalIds,journal,citationCount,pages,volume'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {'error': str(e)}

def get_paper_by_id(paper_id):
    url = (
        f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}'
        f'?fields=title,authors,year,venue,externalIds,journal,citationCount,pages,volume'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {'error': str(e)}

def print_results(label, data):
    print(f'\n{"="*60}')
    print(f'QUERY: {label}')
    print('='*60)
    for p in data.get('data', []):
        print(f"  Title: {p['title']}")
        print(f"  Authors: {[a['name'] for a in p.get('authors', [])]}")
        print(f"  Year: {p.get('year')}  |  Citations: {p.get('citationCount')}")
        print(f"  Venue: {p.get('venue')}")
        print(f"  Journal: {p.get('journal')}")
        print(f"  Volume: {p.get('volume')}  |  Pages: {p.get('pages')}")
        print(f"  ExternalIDs: {p.get('externalIds')}")
        print()

# ---- Paper 1: Leibo et al. 2017 ----
r1 = search_ss('Multi-agent Reinforcement Learning Sequential Social Dilemmas Leibo 2017', limit=4)
print_results('Paper 1: Leibo MARL Social Dilemmas', r1)
time.sleep(1.5)

# ---- Paper 2: Nowak 2006 ----
r2 = search_ss('Five Rules for the Evolution of Cooperation Nowak Science 2006', limit=4)
print_results('Paper 2: Nowak Five Rules Evolution Cooperation', r2)
time.sleep(1.5)

# ---- Paper 3: Reward sharing / incentive alignment MARL ----
r3a = search_ss('reward sharing cooperative multi-agent reinforcement learning emergent cooperation', limit=5)
print_results('Paper 3a: Reward Sharing Cooperative MARL', r3a)
time.sleep(1.5)

r3b = search_ss('Learning to share rewards multi-agent reinforcement learning', limit=5)
print_results('Paper 3b: Learning to share rewards MARL', r3b)
time.sleep(1.5)

r3c = search_ss('intrinsic social motivation reward transfer multi-agent cooperation', limit=5)
print_results('Paper 3c: Intrinsic social motivation reward MARL', r3c)
time.sleep(1.5)

# ---- Paper 4: Independent Q-learners degenerate / Matignon ----
r4a = search_ss('independent reinforcement learners coordination problems Matignon', limit=5)
print_results('Paper 4a: Matignon independent RL coordination', r4a)
time.sleep(1.5)

r4b = search_ss('pathological convergence independent Q-learning multi-agent', limit=5)
print_results('Paper 4b: Pathological convergence independent Q-learning', r4b)
time.sleep(1.5)

r4c = search_ss('Lenient learners cooperative multi-agent reinforcement learning', limit=5)
print_results('Paper 4c: Lenient learners MARL', r4c)
