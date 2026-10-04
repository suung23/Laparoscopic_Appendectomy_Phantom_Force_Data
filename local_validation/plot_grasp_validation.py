"""Plot recorded solver measurements, not animated or prescribed deformation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text())
    rows=json.loads((a.run/'deformation_trace.json').read_text())
    physics=json.loads((a.run/'physics_trace.json').read_text())
    t=np.array([r['step']/30 for r in rows])
    held=np.array([len(r['held']) for r in rows])
    free=np.array([r['nonrigid_rms_m']*1000 for r in rows])
    pull=np.array([r['pull_relative']['nonrigid_rms_m']*1000 if r['pull_relative'] else np.nan for r in rows])
    volumes=np.array([r['tissue_health']['minimum_signed_volume_ratio'] for r in physics])
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[1,2,1.4]})
    fig.suptitle('CT assisted grasp: measured FEM response',x=.1,ha='left',fontsize=17)
    axes[0].step(t,held,where='post',color='#3063ad',linewidth=2)
    axes[0].set_ylabel('Hands holding');axes[0].set_yticks([0,1,2]);axes[0].set_ylim(-.1,2.3)
    axes[1].plot(t,free,label='Non-rigid RMS vs free reset shape',color='#3063ad',linewidth=2)
    axes[1].plot(t,pull,label='Non-rigid RMS vs pre-pull held shape',color='#bc650c',linewidth=2)
    axes[1].set_ylabel('Deformation (mm)');axes[1].legend(loc='upper left',fontsize=9)
    axes[2].plot(np.arange(1,len(volumes)+1)/480,volumes,color='#267751',linewidth=1.5)
    axes[2].axhline(0,color='#ad3333',linestyle='--',linewidth=1)
    axes[2].set_ylabel('Minimum tet\nvolume ratio');axes[2].set_xlabel('Simulation time (s)')
    bounds=report.get('phase_ends')
    if bounds:
        for ax in axes:
            ax.axvspan(bounds[1]/30,bounds[2]/30,color='#6b96d9',alpha=.13)
            ax.axvline(bounds[2]/30,color='#666666',linestyle=':',linewidth=1)
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.text(.1,.015,'Engineering probe | kinematic grasp assistance | uncalibrated material | blue band: pull; dotted line: release',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.035,1,.96])
    fig.savefig(a.run/'mechanics.png',dpi=150)


if __name__=='__main__':main()
