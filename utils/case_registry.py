"""Stable experiment IDs and repository-relative configs."""
CASES = {'case00_small_gridnet': 'config/cases/case00_small_gridnet.yaml',
 'case01_wide_gridnet': 'config/cases/case01_wide_gridnet.yaml',
 'case02_small_mix_skip': 'config/cases/case02_small_mix_skip.yaml',
 'case03_wide_mix_skip': 'config/cases/case03_wide_mix_skip.yaml',
 'case04_small_film': 'config/cases/case04_small_film.yaml',
 'case05_wide_film': 'config/cases/case05_wide_film.yaml',
 'case06_small_t_fla': 'config/cases/case06_small_t_fla.yaml',
 'case07_wide_t_fla': 'config/cases/case07_wide_t_fla.yaml',
 'case08_small_t_bilstm': 'config/cases/case08_small_t_bilstm.yaml',
 'case09_wide_t_bilstm': 'config/cases/case09_wide_t_bilstm.yaml'}

def resolve_case(value):
    matches=[name for name in CASES if name==value or name.split('_')[0]==value]
    if len(matches)!=1:
        raise ValueError('Unknown case: '+value+'; choose case00..case09')
    return matches[0]
