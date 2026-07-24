import xml.etree.ElementTree as ET
from collections import defaultdict
import sys

def load(path):
    t=ET.parse(path); r=t.getroot()
    # module resources (inclusive) from ModuleInformation
    mres={}
    for m in r.findall('ModuleInformation/Module'):
        name=m.find('Name').text
        res=m.find('.//AreaEstimates/Resources')
        def gi(tag):
            e=res.find(tag) if res is not None else None
            return int(e.text) if e is not None and e.text is not None else 0
        mres[name]={'DSP':gi('DSP') or gi('DSP48E'),'LUT':gi('LUT'),'FF':gi('FF'),'BRAM':gi('BRAM_18K')}
    # top total
    top=r.find('AreaEstimates/Resources')
    def gt(tag):
        e=top.find(tag); return int(e.text) if e is not None else 0
    total={'DSP':gt('DSP'),'LUT':gt('LUT'),'FF':gt('FF'),'BRAM':gt('BRAM_18K')}
    return r,mres,total

def category(name):
    n=name.lower()
    if 'einsum_dense' in n: return 'einsum_dense(binary)'
    if n.startswith('einsum_') or 'einsum_ap' in n: return 'einsum(act x act)'
    if 'subln' in n: return 'subln(norm)'
    if 'softmax' in n: return 'softmax'
    if n.startswith('dense_') or n=='dense' or ('dense' in n and 'einsum' not in n): return 'dense(binary)'
    if 'global_pooling' in n or 'pooling' in n: return 'pooling'
    if n.startswith('myproject'): return 'top-glue'
    return 'glue/other'

def census(path,RES='DSP'):
    r,mres,total=load(path)
    top=r.find('RTLDesignHierarchy/TopModule')
    cat_dsp=defaultdict(int)
    per_module_own=defaultdict(int)   # module -> summed own dsp across instances
    inst_count=defaultdict(int)
    running_total=[0]
    def dsp(mod): return mres.get(mod,{}).get(RES,0)
    def walk(inst_elem):
        mod=inst_elem.find('ModuleName').text
        il=inst_elem.find('InstancesList')
        children=il.findall('Instance') if il is not None else []
        own=dsp(mod)-sum(dsp(c.find('ModuleName').text) for c in children)
        cat_dsp[category(mod)]+=own
        per_module_own[mod]+=own
        inst_count[mod]+=1
        running_total[0]+=own
        for c in children: walk(c)
    # top module own
    il=top.find('InstancesList')
    topchildren=il.findall('Instance') if il is not None else []
    topmod=top.find('ModuleName').text
    own_top=mres.get(topmod,{}).get(RES,0)-sum(dsp(c.find('ModuleName').text) for c in topchildren)
    cat_dsp[category(topmod)]+=own_top
    per_module_own[topmod]+=own_top
    running_total[0]+=own_top
    for c in topchildren: walk(c)
    return total,cat_dsp,per_module_own,inst_count,running_total[0]

path=sys.argv[1]
RES=sys.argv[2] if len(sys.argv)>2 else 'DSP'
total,cat_dsp,per_module_own,inst_count,rt=census(path,RES)
print(f'=== {path} ===')
print('profile total DSP:',total['DSP'],' LUT:',total['LUT'],' FF:',total['FF'],' BRAM:',total['BRAM'])
print(f'tree-summed own {RES}:',rt, ' (should equal profile total)')
print(f'--- {RES} by category (own, non-double-counted) ---')
for k,v in sorted(cat_dsp.items(),key=lambda x:-x[1]):
    if v!=0: print(f'  {v:8d}  {k}')
print(f'--- per-module own {RES} (nonzero), with instance count ---')
for mod,v in sorted(per_module_own.items(),key=lambda x:-x[1]):
    if v!=0: print(f'  {v:8d}  x{inst_count[mod]:<3d} {mod}')
