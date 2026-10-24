"""Builds a 21.5 s synthetic vocal with 7 planted problems (for the self-test)."""
import sys, numpy as np, soundfile as sf
np.random.seed(0)
OUT = sys.argv[1] if len(sys.argv) > 1 else 'test_vocal.wav'
from scipy.signal import butter, sosfilt
sr=44100
def tone(midi,dur,cents=0,vib=True):
    t=np.arange(int(sr*dur))/sr
    f=440*2**((midi-69+cents/100)/12)
    fm=f*(1+ (0.006*np.sin(2*np.pi*5.5*t)*np.clip(t/0.3,0,1) if vib else 0))
    ph=2*np.pi*np.cumsum(fm)/sr
    x=sum((1/(k**1.2))*np.sin(k*ph) for k in range(1,12))
    env=np.minimum(1,np.minimum(t/0.04,(dur-t)/0.06)); env=np.clip(env,0,1)
    return 0.18*x*env
seq=[(60,.6),(62,.6),(64,.6),(65,.6),(67,1.0),(None,.2),(67,1.0,40),(65,.6),(64,.6),(62,.6),(60,1.0),(None,.3),
     (64,.8),(67,.8),(72,1.2),(None,.2),(71,.8),(69,.8),(67,1.2),(None,.3),(65,.8),(64,.8),(62,.8),(60,1.5),(None,.3),
     (60,.8),(64,.8),(67,.8),(72,1.6)]
out=[];marks={}
for it in seq:
    if it[0] is None: out.append(np.zeros(int(sr*it[1]))); continue
    c=it[2] if len(it)>2 else 0
    if c: marks['offpitch']=sum(map(len,out))/sr
    out.append(tone(it[0],it[1],c))
y=np.concatenate(out)
# click
i=int(sr*7.3); y[i]+=0.6; y[i+1]-=0.5; marks['click']=7.3
# clipping
a,b=int(sr*10.0),int(sr*10.25); y[a:b]=np.clip(y[a:b]*8,-1,1); marks['clip']=10.0
# glitch: replace 40ms with +7 semitones
i=int(sr*12.6); tt=np.arange(int(sr*0.045))/sr; g=0.18*sum((1/(k**1.2))*np.sin(2*np.pi*k*440*2**((79-69)/12)*tt) for k in range(1,12)); y[i:i+len(g)]=g; marks['glitch']=12.6
# abrupt cut at 15.0: zero the rest of the note until next gap
i=int(sr*15.0); j=i
while j<len(y) and abs(y[j:j+200]).max()>1e-4: j+=200
y[i:j]=0; marks['cut']=15.0
# sibilance burst at 17.2
n=np.random.randn(int(sr*0.12)); sos=butter(4,5500,'hp',fs=sr,output='sos'); n=sosfilt(sos,n); n=n/abs(n).max()*0.8*np.hanning(len(n))
i=int(sr*17.6); y[i:i+len(n)]+=n; marks['sib']=17.6
y=y[:-int(sr*0.5)]  # end cut while sounding
sf.write(OUT,y.astype(np.float32),sr)
print(marks, len(y)/sr)
