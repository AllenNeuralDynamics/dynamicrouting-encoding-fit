import numpy as np

session_id = '620263_2022-07-26'
with open(f'{session_id}_full_model.npz', 'wb') as f:
    np.savez(
        f,
        design_matrix=np.full((5, 5), 1.1),
        spike_rate=np.full((5, 5), 1.1),
        params={
            'session_id': session_id,
            'model_name': 'full_model',
        },
    )