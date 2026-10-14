import argparse,json,urllib.request,time,statistics,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('phase',choices=['draft','prefill']);p.add_argument('--label',default='baseline');p.add_argument('--policies',default='auto,fcost7:noisy,f7,fnc7:0.15,fnc7:0.5');a=p.parse_args()
url='http://127.0.0.1:8888'
path=Path('/tmp/glm-perf-'+a.label+'-'+a.phase+'.jsonl')
def health():
 with urllib.request.urlopen(url+'/health',timeout=10) as r:return json.load(r)
def avail():
 return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1048576
def run(name,prompt,tokens,policy='auto',draft=True,rep=0):
 before=health()
 if before['busy']:raise RuntimeError('Server is handling another request; stopping benchmark')
 body={'model':'GLM-5.3-Flash-EXL3','messages':[{'role':'user','content':prompt}], 'max_tokens':tokens,'temperature':0,'draft':draft,'tf_policy':policy,'chat_template_kwargs':{'enable_thinking':False}}
 stop=threading.Event();mem=[]
 def monitor():
  while not stop.wait(.2):mem.append(avail())
 thread=threading.Thread(target=monitor,daemon=True);thread.start();t=time.monotonic()
 try:
  req=urllib.request.Request(url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=600) as r:r=json.load(r)
 finally:stop.set();thread.join()
 after=health();u=r['usage'];s=r['tensorfold'];n=u['completion_tokens']
 assert after['completion_tokens_total']-before['completion_tokens_total']==n,'Concurrent request contaminated benchmark'
 result={'label':a.label,'name':name,'rep':rep,'policy':policy,'draft':draft,'prompt_tokens':u['prompt_tokens'],'completion_tokens':n,'cached_tokens':s.get('cached',0),'prefill_s':s['prefill_s'],'decode_s':s['decode_s'],'decode_tps':(n-1)/s['decode_s'] if s['decode_s'] else None,'prefill_tps':(u['prompt_tokens']-s.get('cached',0))/s['prefill_s'],'token_sha':s.get('token_sha'),'rounds':s.get('rounds'),'accepted':s.get('accepted'),'drafted':s.get('drafted'),'min_available_gib':min(mem+[avail()]),'wall_s':time.monotonic()-t}
 with path.open('a') as f:f.write(json.dumps(result)+'\n')
 print(json.dumps(result),flush=True);return result
if a.phase=='draft':
 jobs=[('code','Write a complete Python standard-library implementation of an LRU cache with TTL, thread safety, a fake clock, and unittest tests. Explain the invariants in comments.'),('prose','Write a detailed practical guide to maintaining a bicycle, including cleaning, inspecting brakes, adjusting gears, and fixing a flat. Use complete explanatory paragraphs.')]
 hashes={}
 for rep in range(3):
  policies=a.policies.split(',')
  if rep:policies.reverse()
  for policy in policies:
   for name,prompt in jobs:
    r=run(name,prompt,512,policy,rep=rep)
    if name in hashes:assert hashes[name]==r['token_sha'],(name,'output differs across policies')
    else:hashes[name]=r['token_sha']
else:
 for lines in (480,1800):
  prompt='\n'.join(f'Record {i}: warehouse has {i%97} blue bolts and {i%83} green nuts.' for i in range(lines))+'\nSummarize the record format in one sentence.'
  for rep in range(3):run('records-'+str(lines),prompt,16,draft=False,rep=rep)
