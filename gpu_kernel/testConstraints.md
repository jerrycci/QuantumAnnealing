## config:
1. beta = 0,01~100
2. \# Monte Carlo step = 100,000

### Total Qubo Formula:
```
result = h2d + panelty * (c12d/128**2 + 2*c22d + c32d/45**2 + c42d/273**2 + 2*c452d/273**2 + c52d/110**2 + 2*c62d)
```
## result
1. c1_check = 121.0
2. c3_check = 272.0
3. c4_check = 52.0
4. c5_check = 3.0
5. c12d = 5.0
6. c22d = 0.0
7. c32d = 0.0
8. c42d = 1.0
9.  c452d = 248.0
10. c52d = 3.25
11. c62d = 0.0

Final MLB throughput = -0.85278

Final total energy = -0.12856

Total time = 10.65575