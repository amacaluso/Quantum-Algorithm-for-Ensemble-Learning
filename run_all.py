from Utils import *
create_dir('output')

seed = 123
np.random.seed(seed)

runs = 20
data = pd.DataFrame()

for i in range(runs):
    x1 = uniform(-10,10,2)
    x2 = uniform(-10,10,2)
    x_test = uniform(-10,10,2)

    row = pd.Series(np.concatenate((x1, x2, x_test)))
    data = data.append(row, ignore_index=True)

data.columns = ['x11', 'x12', 'x21', 'x22', 'x1_test', 'x2_test']
data.to_csv('output/data.csv', index=False)

data

# Execution on real device
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends()
backend_16 = provider.get_backend('ibmq_16_melbourne')
backend_5 = provider.get_backend('ibmq_rome')

data_out = pd.DataFrame()
i = 0
for index, rows in data.iterrows():
    # Extract data
    x1 = np.array([rows.x11, rows.x12])
    x2 = np.array([rows.x21, rows.x21])
    x_test = np.array([rows.x1_test, rows.x2_test])

    # Compute the average classically of classical swap test
    cAVG = classic_ensemble(x1, x2, x_test)

    # Swap test using x1 as training on simulator
    qc_x1 = quantum_swap_test(x1, x_test)
    m1 = exec_simulator(qc_x1)
    r1 = retrieve_proba(m1)

    # Swap test using x2 as training on simulator
    qc_x2 = quantum_swap_test(x2, x_test)
    m2 = exec_simulator(qc_x2)
    r2 = retrieve_proba(m2)

    # Compute the average classically
    r_avg = np.mean([r1[0], r2[0]])

    # Compute the average using quantum ensemble algorithm on simulator
    qc = quantum_ensemble(x1, x2, x_test)
    r = exec_simulator(qc)
    r_ens = retrieve_proba(r)

    # Using real devices

    # Swap test using x1 as training on real device
    qc_x1_real = quantum_swap_test(x1, x_test)
    r_x1_real = run_real_device(qc_x1_real, backend_5)
    r1_rl = retrieve_proba(r_x1_real)

    # Swap test using x2 as training on real device
    qc_x2_real = quantum_swap_test(x2, x_test)
    r_x2_real = run_real_device(qc_x2_real, backend_5)
    r2_rl = retrieve_proba(r_x2_real)

    # Compute the average classically
    r_avg_real = np.mean([r1_rl[0], r2_rl[0]])

    # Compute the average using quantum ensemble algorithm on real device
    qc = quantum_ensemble(x1, x2, x_test)
    qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)

    r_ens_real = run_real_device(qc, backend_16)
    r_ens_rl = retrieve_proba(r_ens_real)

    row = [cAVG, r1[0], r2[0], r_avg, r_ens[0], r1_rl[0], r2_rl[0], r_avg_real, r_ens_rl[0]]

    row = pd.Series(row)

    print(i)
    i = i + 1

    data_out = data_out.append(row, ignore_index=True)

data_out.columns = ['cAVG', 'qx1_sim', 'qx2_sim', 'qAVG_sim', 'qEns_sim',
                    'qx1_real', 'qx2_real', 'qAVG_real', 'qEns_real']

data_out.to_csv('output/data_out_final.csv', index=False)
