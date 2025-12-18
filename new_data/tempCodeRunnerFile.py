'X': el.X,                              # 电负性
                    'Radius': el.atomic_radius,             # 原子半径
                    'Mass': el.atomic_mass,                 # 质量
                    'Ionization': el.ionization_energy,     # 电离能 (关键!)
                    'Affinity': el.electron_affinity,       # 电子亲和能 (关键!)
                    'MeltingT': el.melting_point,           # 熔点 (反映键强)
                    'Valence': float(el.nvalence_electrons) # 价电子数