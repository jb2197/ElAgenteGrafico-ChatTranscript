import numpy as np, math, json, os, textwrap
from pyscf.lib import chkfile

# Data transcribed from workflow outputs
solvents = {
    'water': [
        {
            'label':'W1','source_conf':'set1','degen':27,
            'energy':-1068.9139220838401,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/5c043c5f-8228-498a-971f-51919a74a9ff.chk',
            'exc':[0.11165340176496104,0.14925840030810608,0.15013574212508,0.15742894717938155,0.16451037219347692,0.18219673479087806,0.18905365379699543,0.19115273940576347,0.19371891028413707,0.19981896339969432],
            'osc':[1.1041044493568202,9.359673904579097e-7,0.2571369668350875,7.113008559435815e-7,0.4073503719663646,0.0001189875309559085,0.038168762094930864,0.00001879645654328548,0.012127242650294101,0.11175594367977194],
        },
        {
            'label':'W2','source_conf':'set2','degen':21,
            'energy':-1068.9145084529546,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/4451cdc9-3f0e-40bb-b92f-56d5717f4aa5.chk',
            'exc':[0.11044005956356535,0.14768303090997242,0.1505401465778259,0.15732732483629203,0.1632628862284317,0.18201726454776426,0.18832967581703042,0.19139019999002074,0.19383467688571301,0.1980593003875638],
            'osc':[1.0927062179852398,0.00002303902028386645,0.23416216752083152,1.6429376622355842e-6,0.4066570308034576,0.00011765522071100795,0.02858425747586517,0.000039442510708503976,0.02388291758862631,0.11796160326279943],
        },
        {
            'label':'W3','source_conf':'set3','degen':25,
            'energy':-1068.9135623280033,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/f7bfa87b-4c67-4b94-9e19-44b3cbe5993f.chk',
            'exc':[0.10382547443932322,0.15007990808938132,0.15387149832256605,0.15692428345444406,0.16053287093449145,0.18165241947211933,0.18790742527335189,0.1914833889676062,0.19368322669774055,0.19541514682324912],
            'osc':[0.9765332485916094,0.31856297451655824,0.0000811216083806974,2.067913061691614e-6,0.2947239625126685,0.0001092201448113846,0.005021085572606134,0.018584049775858088,0.00002794652318722519,0.052994184122355466],
        },
        {
            'label':'W4','source_conf':'set4','degen':50,
            'energy':-1068.9120811915386,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/6e877313-0a99-4924-829d-982df7362b04.chk',
            'exc':[0.11256398803253137,0.14989473414754184,0.15052023305222748,0.15747700362743058,0.16514438241790683,0.18230169915791475,0.18968331634268132,0.19108803727875648,0.19365018946417067,0.19855228522176502],
            'osc':[1.050572254787033,0.27517884846951,0.012240513686107887,0.0001263454002538377,0.3538559647356362,0.00013228773326954358,0.019558651859163834,0.0013735746767390982,0.009954345313543115,0.08074117213165294],
        },
        {
            'label':'W5','source_conf':'set5','degen':1,
            'energy':-1068.9112657017063,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/5c2f7e18-7a57-470e-a37a-36ae13171b57.chk',
            'exc':[0.10549528299499616,0.1495573298243589,0.15574206499099114,0.15706571980270992,0.1622412988645597,0.1818995249371809,0.18915491698409934,0.19152062768659991,0.1928205179362775,0.19589733467752085],
            'osc':[0.9506376228569869,0.3153463080861129,0.00007288616682957662,0.000024442276429666498,0.31419120456750643,0.0001199843635768272,0.020088652159769863,0.02189413127678482,0.00006127969442112765,0.033267116774170775],
        },
    ],
    'heptane': [
        {
            'label':'H1','source_conf':'set1','degen':48,
            'energy':-1068.9325366873775,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/5255da87-821d-4a18-829f-404ad9c10436.chk',
            'exc':[0.10604691415162275,0.13590366747644114,0.15573445341599623,0.1561599255697759,0.16384703394110783,0.17858462097082678,0.18878802507495546,0.18988954829345378,0.19790619837722137,0.19941005210010526],
            'osc':[1.0843623709793664,8.407447373586043e-6,7.588037915367605e-6,0.07344738026821249,0.5532862021981821,0.0001134279593317859,0.005458844420944319,0.000016239644287404953,0.1371784865167299,0.023702993527886976],
        },
        {
            'label':'H2','source_conf':'set2','degen':1,
            'energy':-1068.9318243704279,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/8b405710-ea71-484d-b0bb-ef0306b04e40.chk',
            'exc':[0.10684915562661318,0.1372967113940808,0.15584371011012926,0.1561601848916888,0.1643950028475911,0.1787793866558809,0.18941811325821947,0.18954364191521603,0.1986562132160532,0.20010943145928775],
            'osc':[1.0926509765039571,0.000021360819198877596,0.00023537818672653724,0.10754538019562994,0.5462086224349045,0.00011528853113864986,0.00010939452139431283,0.010721650014060512,0.029414273110741285,0.11743142639701418],
        },
        {
            'label':'H3','source_conf':'set3','degen':9,
            'energy':-1068.9306897711497,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/d6ac1d2b-aa48-478c-b98b-ece85796b56e.chk',
            'exc':[0.09946682281535846,0.13970077270448847,0.15513817300308008,0.1558168844673816,0.16142228940550374,0.17798224464740306,0.18859465653529858,0.19170654608698326,0.19277843087060975,0.19796764773777875],
            'osc':[0.9839091198872474,0.00001595452740247048,0.0018090612139904545,0.16012068469059815,0.40991053592448135,0.00010300047370295075,0.0006194313991643946,0.00003056051319902887,0.002619266583436637,0.07170592690923451],
        },
        {
            'label':'H4','source_conf':'set4','degen':13,
            'energy':-1068.929757359592,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/56716c6f-e328-408c-8b69-bee490d8b625.chk',
            'exc':[0.10714647481498704,0.13766292641935812,0.15575304097504566,0.15624910240163478,0.164469826447149,0.1788241180443361,0.1891716398548483,0.19015939216691674,0.1977634125226641,0.1993099286798897],
            'osc':[1.050439086892345,0.0019311208974434873,0.027334421251662848,0.10655028471686481,0.5013530975147844,0.00012581585808773822,0.0017984852682297024,0.004843244443408366,0.06132373108211604,0.03145906773801417],
        },
        {
            'label':'H5','source_conf':'set5','degen':2,
            'energy':-1068.9196917244496,
            'chk':'/scratch/jiarubai/akg4pyscf/domains/pyscf/chkfiles/9a17054c-dfc4-472a-8749-8ee31dca3215.chk',
            'exc':[0.10248249049294457,0.14887290996718017,0.15247656727013334,0.15608245780929758,0.16137422644839644,0.17970332281220042,0.18039260255982797,0.1854147026442239,0.18859402291051588,0.19422046340916904],
            'osc':[0.43371390214347005,0.09998894867278707,0.3360819863140037,0.014749732219060705,0.04929990908293436,0.00125508309823797,0.04777976506384496,0.009928141253741182,0.04449448699245854,0.10843322674528164],
        },
    ]
}

# Load coordinates and symbols from chkfiles
for solv, lst in solvents.items():
    for d in lst:
        mol = chkfile.load_mol(d['chk'])
        d['coords'] = mol.atom_coords(unit='Angstrom')
        d['symbols'] = [mol.atom_symbol(i) for i in range(mol.natm)]

heavy_idx = [i for i,s in enumerate(solvents['water'][0]['symbols']) if s != 'H']

def kabsch_rmsd(P, Q):
    # P,Q shape (n,3)
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1,1,d])
    U = V @ D @ Wt
    P_rot = Pc @ U
    diff = P_rot - Qc
    return np.sqrt((diff*diff).sum()/len(P))

Eh_to_kcal = 627.509474
kB_kcal = 0.00198720425864083
T=298.15

# Pairwise RMSD and energy differences within each solvent
for solv, lst in solvents.items():
    print('\nSOLVENT', solv)
    energies = np.array([d['energy'] for d in lst])
    rel = (energies-energies.min())*Eh_to_kcal
    for d,r in zip(lst,rel):
        d['rel_kcal'] = float(r)
        print(d['label'], 'rel_kcal', round(r,4))
    print('Pairwise heavy-atom RMSD / dE')
    for i in range(len(lst)):
        for j in range(i+1,len(lst)):
            rmsd = kabsch_rmsd(lst[i]['coords'][heavy_idx], lst[j]['coords'][heavy_idx])
            de = abs(lst[i]['energy']-lst[j]['energy'])*Eh_to_kcal
            print(lst[i]['label'], lst[j]['label'], 'dE=', round(de,4), 'kcal/mol', 'RMSD=', round(rmsd,4), 'A')
            lst[i].setdefault('pairs',{})[lst[j]['label']] = (de, rmsd)
    
# Simple clustering of duplicates: dE <= 0.05 kcal/mol and RMSD <= 0.15 A
for solv, lst in solvents.items():
    parent = list(range(len(lst)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra
    for i in range(len(lst)):
        for j in range(i+1,len(lst)):
            rmsd = kabsch_rmsd(lst[i]['coords'][heavy_idx], lst[j]['coords'][heavy_idx])
            de = abs(lst[i]['energy']-lst[j]['energy'])*Eh_to_kcal
            if de <= 0.05 and rmsd <= 0.15:
                union(i,j)
    groups = {}
    for i,d in enumerate(lst):
        groups.setdefault(find(i), []).append(i)
    print('\nClusters for',solv)
    for g, idxs in groups.items():
        print([lst[i]['label'] for i in idxs])
        # choose lowest-energy representative; sum degeneracies
        rep = min(idxs, key=lambda i: lst[i]['energy'])
        total_degen = sum(lst[i]['degen'] for i in idxs)
        for i in idxs:
            lst[i]['cluster_rep'] = lst[rep]['label']
            lst[i]['cluster_total_degen'] = total_degen
    
# Build merged minima with summed degeneracy and lowest-energy representative spectra
merged = {}
for solv, lst in solvents.items():
    reps = {}
    for d in lst:
        reps.setdefault(d['cluster_rep'], {'members':[], 'degen':0})
        reps[d['cluster_rep']]['members'].append(d)
        reps[d['cluster_rep']]['degen'] += d['degen']
    merged[solv] = []
    for rep_label, info in reps.items():
        rep = min(info['members'], key=lambda x: x['energy'])
        m = {k:v for k,v in rep.items() if k not in ('coords','symbols','pairs')}
        m['merged_degen'] = info['degen']
        m['members'] = [x['label'] for x in info['members']]
        merged[solv].append(m)
    merged[solv].sort(key=lambda x: x['energy'])

for solv, lst in merged.items():
    Emin = min(d['energy'] for d in lst)
    weights = np.array([d['merged_degen']*math.exp(-(d['energy']-Emin)*Eh_to_kcal/(kB_kcal*T)) for d in lst])
    weights /= weights.sum()
    print('\nMerged weights for',solv)
    for d,w in zip(lst,weights):
        d['final_weight'] = float(w)
        d['rel_kcal'] = float((d['energy']-Emin)*Eh_to_kcal)
        print(d['label'], d['members'], 'deg', d['merged_degen'], 'rel_kcal', round(d['rel_kcal'],4), 'w', round(float(w),4))

# Save json summary for later plotting/reporting
out = {'merged': merged, 'solvents': {k:[{kk:vv for kk,vv in d.items() if kk not in ('coords','symbols','pairs')} for d in v] for k,v in solvents.items()}}
open('/tmp/ensemble_summary.json','w').write(json.dumps(out, indent=2))
print('\nWrote /tmp/ensemble_summary.json')
