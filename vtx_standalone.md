
Build with
```bash
nvcc ./vtx_standalone.cu --std=c++11 -o vtx_standalone ./tracker.o ./vort.o ./minions.o
```

Run with:
```bash
vtx_standalone <start_timestep> <end_timestep> <2d mask radius> <increment>
```
