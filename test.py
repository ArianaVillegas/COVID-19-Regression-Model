import os
import json

import numpy as np

from model import apply_models


def test(path, covid_region, regions, labels, alpha_values, iterations_values, error_methods, regularization,
         opt_methods, train_size):
    if not os.path.exists('/'.join(path)):
        os.makedirs('/'.join(path))

    for region, name in zip(covid_region, regions):
        path.append(name)

        # Preprocess data
        test_results = {}
        covid_test = region.groupby(by=['date_num', 'lon', 'lat']).sum()
        index = np.array(list(map(lambda t: list(t), covid_test.index)))
        covid_test['date_num'] = index[:, 0]
        covid_test['lon'] = index[:, 1]
        covid_test['lat'] = index[:, 2]
        covid_x = covid_test[['date_num', 'lon', 'lat']]
        covid_x_label = {}
        for _, row in region.iterrows():
            covid_x_label[row['date_num']] = row['date']

        for label in labels:
            path.append(label)

            # Select label
            covid_y = covid_test[label]

            for alpha in alpha_values:
                path.append('alpha_'+str(alpha))

                for itr in iterations_values:
                    path.append('itr_'+str(itr))

                    for err in error_methods:
                        path.append(err)

                        for reg in regularization:
                            path.append(reg)

                            for opt in opt_methods:
                                path.append(opt)

                                print('/'.join(path))
                                os.makedirs('/'.join(path))

                                size = int(len(covid_test) * train_size)
                                apply_models(covid_x, covid_x_label, covid_y, size, test_results, itr, err, alpha,
                                             reg, opt, '/'.join(path))

                                path.pop()
                            path.pop()
                        path.pop()
                    path.pop()
                path.pop()
            path.pop()
        path.pop()

        with open('test_results_'+name+'.txt', 'w') as file:
            file.write(json.dumps(test_results))
