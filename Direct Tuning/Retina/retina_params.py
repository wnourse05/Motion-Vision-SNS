import pickle

params = {'g_m': 1.0,
          'c_m': 5.0,
          'v_r': 0.0}
pickle.dump(params, open('retina_params.p', 'wb'))