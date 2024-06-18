import lmfit
import numpy as np

n = 60
lhs_n = 45
maxSize = 10000
days = 1096


#Kill it when it goes below e-9
#Include detection limits
params = lmfit.Parameters()
params.add('g', value=0.04, min=0.01, max=0.06, vary=True)
params.add('gDistr', value=0.2, min=0, max=0.5, vary=False)
params.add('d', value=0.1, min=0.03, max=0.25, vary=True)
params.add('dDistr', value=0.2, min=0, max=0.5, vary=False)
params.add('r', value=0.04, min=0, max=0.05, vary=True)
params.add('rDistr', value=0.2, min=0, max=0.5, vary=False)

medDayScan = 40
dayNoise = 10
scanNoise = 0.01

ideal = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.994219653179191, 0.994219653179191, 0.994219653179191, 0.994219653179191, 0.994219653179191, 0.988439306358382, 0.988439306358382, 0.988439306358382, 0.982658959537572, 0.982658959537572, 0.982658959537572, 0.971098265895954, 0.971098265895954, 0.965317919075145, 0.953757225433526, 0.953757225433526, 0.947976878612717, 0.947976878612717, 0.947976878612717, 0.947976878612717, 0.942196531791908, 0.942196531791908, 0.936416184971098, 0.930635838150289, 0.92485549132948, 0.919075144508671, 0.919075144508671, 0.913294797687861, 0.913294797687861, 0.913294797687861, 0.913294797687861, 0.913294797687861, 0.913294797687861, 0.907514450867052, 0.907514450867052, 0.907514450867052, 0.907514450867052, 0.895953757225434, 0.895953757225434, 0.895953757225434, 0.895953757225434, 0.895953757225434, 0.884393063583815, 0.884393063583815, 0.884393063583815, 0.884393063583815, 0.872832369942196, 0.867052023121387, 0.867052023121387, 0.861271676300578, 0.861271676300578, 0.855491329479769, 0.84971098265896, 0.838150289017341, 0.832369942196532, 0.832369942196532, 0.826589595375723, 0.815028901734104, 0.815028901734104, 0.815028901734104, 0.815028901734104, 0.815028901734104, 0.815028901734104, 0.809248554913295, 0.797687861271676, 0.797687861271676, 0.797687861271676, 0.797687861271676, 0.797687861271676, 0.791907514450867, 0.791907514450867, 0.780346820809249, 0.763005780346821, 0.763005780346821, 0.763005780346821, 0.763005780346821, 0.763005780346821, 0.751445086705202, 0.745664739884393, 0.728323699421965, 0.722543352601156, 0.716763005780347, 0.716763005780347, 0.716763005780347, 0.716763005780347, 0.710982658959538, 0.705202312138728, 0.69364161849711, 0.69364161849711, 0.69364161849711, 0.687861271676301, 0.682080924855491, 0.676300578034682, 0.670520231213873, 0.670520231213873, 0.653179190751445, 0.641618497109827, 0.630057803468208, 0.61849710982659, 0.606936416184971, 0.606936416184971, 0.606936416184971, 0.601156069364162, 0.595375722543353, 0.595375722543353, 0.595375722543353, 0.589595375722543, 0.589595375722543, 0.572254335260116, 0.572254335260116, 0.572254335260116, 0.560693641618497, 0.560693641618497, 0.554913294797688, 0.554913294797688, 0.554913294797688, 0.554913294797688, 0.554913294797688, 0.554913294797688, 0.554913294797688, 0.554913294797688, 0.554913294797688, 0.549132947976879, 0.549132947976879, 0.549132947976879, 0.543352601156069, 0.543352601156069, 0.543352601156069, 0.543352601156069, 0.531791907514451, 0.531791907514451, 0.531791907514451, 0.531791907514451, 0.531791907514451, 0.531791907514451, 0.531791907514451, 0.531791907514451, 0.520231213872832, 0.514450867052023, 0.514450867052023, 0.514450867052023, 0.508670520231214, 0.502890173410405, 0.491329479768786, 0.491329479768786, 0.491329479768786, 0.491329479768786, 0.485549132947977, 0.479768786127168, 0.473988439306358, 0.473988439306358, 0.468208092485549, 0.456647398843931, 0.450867052023121, 0.450867052023121, 0.445086705202312, 0.439306358381503, 0.433526011560694, 0.433526011560694, 0.416184971098266, 0.410404624277457, 0.404624277456647, 0.393063583815029, 0.38728323699422, 0.38150289017341, 0.38150289017341, 0.38150289017341, 0.375722543352601, 0.364161849710983, 0.364161849710983, 0.358381502890173, 0.346820809248555, 0.346820809248555, 0.335260115606936, 0.335260115606936, 0.329479768786127, 0.317919075144509, 0.317919075144509, 0.317919075144509, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.312138728323699, 0.30635838150289, 0.300578034682081, 0.300578034682081, 0.300578034682081, 0.300578034682081, 0.300578034682081, 0.294797687861272, 0.294797687861272, 0.294797687861272, 0.294797687861272, 0.289017341040462, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.283236994219653, 0.277456647398844, 0.277456647398844, 0.277456647398844, 0.277456647398844, 0.277456647398844, 0.277456647398844, 0.277456647398844, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.271676300578035, 0.265895953757225, 0.265895953757225, 0.265895953757225, 0.265895953757225, 0.265895953757225, 0.265895953757225, 0.265895953757225, 0.260115606936416, 0.260115606936416, 0.260115606936416, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.254335260115607, 0.248554913294798, 0.248554913294798, 0.242774566473988, 0.242774566473988, 0.242774566473988, 0.242774566473988, 0.242774566473988, 0.242774566473988, 0.242774566473988, 0.236994219653179, 0.236994219653179, 0.236994219653179, 0.236994219653179, 0.236994219653179, 0.236994219653179, 0.23121387283237, 0.23121387283237, 0.23121387283237, 0.23121387283237, 0.23121387283237, 0.23121387283237, 0.225433526011561, 0.219653179190751, 0.219653179190751, 0.219653179190751, 0.219653179190751, 0.213872832369942, 0.208092485549133, 0.208092485549133, 0.208092485549133, 0.208092485549133, 0.208092485549133, 0.208092485549133, 0.208092485549133, 0.202312138728324, 0.202312138728324, 0.196531791907514, 0.196531791907514, 0.190751445086705, 0.190751445086705, 0.190751445086705, 0.190751445086705, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.184971098265896, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.173410404624277, 0.167630057803468, 0.161849710982659, 0.161849710982659, 0.161849710982659, 0.161849710982659, 0.161849710982659, 0.15606936416185, 0.15606936416185, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.15028901734104, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.144508670520231, 0.138728323699422, 0.138728323699422, 0.138728323699422, 0.138728323699422, 0.138728323699422, 0.138728323699422, 0.138728323699422, 0.132947976878613, 0.127167630057803, 0.127167630057803, 0.127167630057803, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.121387283236994, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.109826589595376, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.104046242774566, 0.0982658959537572, 0.0982658959537572, 0.0982658959537572, 0.092485549132948, 0.0867052023121387, 0.0867052023121387, 0.0867052023121387, 0.0867052023121387, 0.0867052023121387, 0.0867052023121387, 0.0867052023121387, 0.0867052023121387, 0.0809248554913295, 0.0751445086705202, 0.0751445086705202, 0.0751445086705202, 0.069364161849711, 0.069364161849711, 0.069364161849711, 0.069364161849711, 0.069364161849711, 0.0635838150289017, 0.0635838150289017, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0578034682080925, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.0520231213872832, 0.046242774566474, 0.046242774566474, 0.046242774566474, 0.046242774566474, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0404624277456647, 0.0346820809248555, 0.0346820809248555, 0.0346820809248555, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.0289017341040462, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.023121387283237, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0173410404624277, 0.0115606936416185, 0.0115606936416185, 0.0115606936416185, 0.0115606936416185, 0.0115606936416185, 0.0115606936416185, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0.00578034682080925, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

