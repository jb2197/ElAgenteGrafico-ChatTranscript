import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

summary = json.load(open('/tmp/ensemble_summary.json'))
merged = summary['merged']

Eh_to_eV = 27.211386245988
nm_factor = 1239.8419843320026

# Convert lists from json (may be strings? no)
for solv in merged:
    for d in merged[solv]:
        d['exc'] = list(map(float, d['exc']))
        d['osc'] = list(map(float, d['osc']))
        d['final_weight'] = float(d['final_weight'])
        d['rel_kcal'] = float(d['rel_kcal'])

E_grid = np.linspace(2.0, 6.2, 2400)  # eV ~ 620-200 nm
sigma = 0.15  # eV

def broaden(exc_h, osc, weight=1.0):
    exc_eV = np.array(exc_h) * Eh_to_eV
    osc = np.array(osc)
    y = np.zeros_like(E_grid)
    for e, f in zip(exc_eV, osc):
        y += weight * f * np.exp(-0.5*((E_grid-e)/sigma)**2)
    return y

spectra = {}
main = {}
for solv, lst in merged.items():
    y = np.zeros_like(E_grid)
    for d in lst:
        y += broaden(d['exc'], d['osc'], d['final_weight'])
    spectra[solv] = y
    main_d = max(lst, key=lambda x: x['final_weight'])
    main[solv] = {'label': main_d['label'], 'y': broaden(main_d['exc'], main_d['osc'], 1.0)}

wavelength = nm_factor / E_grid
# sort for plotting in increasing wavelength left-to-right? use increasing nm
idx = np.argsort(wavelength)
wl = wavelength[idx]

# Find maxima in 200-500 nm region
report = {}
for solv in spectra:
    y = spectra[solv][idx]
    mask = (wl >= 200) & (wl <= 500)
    i = np.argmax(y[mask])
    wlm = wl[mask][i]
    report[solv] = {'ensemble_lambda_max_nm': float(wlm), 'ensemble_peak_intensity': float(y[mask][i])}
    ym = main[solv]['y'][idx]
    i2 = np.argmax(ym[mask])
    report[solv]['main_label'] = main[solv]['label']
    report[solv]['main_lambda_max_nm'] = float(wl[mask][i2])
    report[solv]['main_peak_intensity'] = float(ym[mask][i2])

colors={'water':'#1f77b4','heptane':'#d62728'}
fig, axes = plt.subplots(2,1, figsize=(8,8), sharex=True)
for solv in ['water','heptane']:
    axes[0].plot(wl, spectra[solv][idx], color=colors[solv], lw=2.0, label=f"{solv.title()} ensemble")
    axes[1].plot(wl, main[solv]['y'][idx], color=colors[solv], lw=2.0, label=f"{solv.title()} dominant conformer ({main[solv]['label']})")
for ax, title in zip(axes, ['Boltzmann-weighted ensemble spectrum', 'Dominant-conformer spectrum']):
    ax.set_xlim(200, 500)
    ax.set_ylabel('Arb. intensity')
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
axes[1].set_xlabel('Wavelength / nm')
fig.suptitle('Implicit-solvent shift comparison (wB97X-D4/def2-SVP, SMD, Gaussian broadening σ=0.15 eV)')
fig.tight_layout(rect=[0,0,1,0.97])
combined='/tmp/solvent_shift_comparison.svg'
fig.savefig(combined)
plt.close(fig)

# individual plots too
for name, data_dict, title in [
    ('/tmp/ensemble_spectra.svg', spectra, 'Boltzmann-weighted ensemble spectrum'),
    ('/tmp/dominant_conformer_spectra.svg', {k:v['y'] for k,v in main.items()}, 'Dominant-conformer spectrum')
]:
    plt.figure(figsize=(8,4.5))
    for solv in ['water','heptane']:
        plt.plot(wl, data_dict[solv][idx], color=colors[solv], lw=2.0, label=solv.title())
    plt.xlim(200,500)
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Arb. intensity')
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

print(json.dumps(report, indent=2))
print('Saved:', combined, '/tmp/ensemble_spectra.svg', '/tmp/dominant_conformer_spectra.svg')
