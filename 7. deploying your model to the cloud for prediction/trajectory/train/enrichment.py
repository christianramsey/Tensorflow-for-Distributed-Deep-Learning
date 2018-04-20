import Algorithmia

input = {
  "lat": "50.2111",
  "lon": "134.1233"
}
client = Algorithmia.client('simNM072YShgdE1J61tme12qhs31')
algo = client.algo('Gaploid/Elevation/0.3.6')
print(algo.pipe(input))