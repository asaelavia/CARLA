import itertools
import operator
import pickle
import re

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from z3 import *
import lime.lime_tabular

filename = 'adult_clean.csv'
constraints_name = 'adult_good_adcs_test.txt'
categorical_column_names = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                            'native_country']
fixed_column_names = ['age', 'race', 'sex']
random_state = 0
num_of_rows_to_run = 10
num_of_cfs = 4
ops = {'==': operator.eq, '!=': operator.ne, '<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge}
FIXED_FEAT = ['age', 'race', 'sex']
CONT_FEAT = ['age', 'edunum', 'hours_per_week']


# def generate_perturbed_samples(instance, dataset, num_samples=1000):
#     mean = dataset.mean()
#     std = dataset.std()
#
#     # Generate random perturbations
#     perturbations = np.random.normal(0, 1, size=(num_samples, len(instance)))
#
#     # Convert perturbations to a DataFrame
#     perturbations_df = pd.DataFrame(perturbations, columns=dataset.columns)
#
#     # Scale perturbations by the dataset's standard deviation
#     scaled_perturbations = perturbations_df * std
#
#     # Add perturbations to the instance
#     instance_df = pd.DataFrame([instance] * num_samples, columns=dataset.columns)
#     perturbed_samples = instance_df + scaled_perturbations
#
#     # Clip values to be within the range of the original dataset
#     min_vals = dataset.min()
#     max_vals = dataset.max()
#     perturbed_samples = perturbed_samples.clip(lower=min_vals, upper=max_vals, axis=1)
#
#     # Round categorical features to integers
#     for col in dataset.columns:
#         if col not in CONT_FEAT:
#             perturbed_samples[col] = np.round(perturbed_samples[col]).astype(int)
#
#     return perturbed_samples.values
def diversity(vars, found_points, x_train):
    if len(found_points) == 0:
        return 0
    total_distance = 0
    for i in range(len(found_points)):
        for k in range(i + 1, len(found_points)):
            for j in non_cat_col_indices:
                median = x_train.iloc[:, j].median()
                med = np.median([abs(x - median) for x in x_train.iloc[:, j]])
                total_distance += (abs(found_points[i][j] - found_points[k][j])) / med
            total_distance += np.sum([(found_points[i][j] != found_points[k][j]) for j in cat_col_indices])
    for i in range(len(found_points)):
        for var in vars:
            j = x_train.columns.get_loc(var)
            if j in non_cat_col_indices:
                median = x_train.iloc[:, j].median()
                med = np.median([abs(x - median) for x in x_train[var]])
                total_distance += Sum((Abs(found_points[i][j] - vars[var])) / med)
            else:
                total_distance += Sum(vars[var] != found_points[i][j])
    return total_distance / ((len(found_points) * (len(found_points) + 1)) / 2)


def generate_perturbed_samples(instance, dataset, num_samples=10000):
    # Calculate mean and std for continuous features
    cont_mean = dataset[CONT_FEAT].mean()
    cont_std = dataset[CONT_FEAT].std()

    # Create a DataFrame with repeated instances
    perturbed_samples = pd.DataFrame([instance] * num_samples, columns=dataset.columns)

    for col in dataset.columns:
        if col not in CONT_FEAT:
            # For categorical features, randomly sample from the unique values in the dataset
            if np.random.random() > 0.5:
                unique_values = dataset[col].unique()
                perturbed_samples[col] = np.random.choice(unique_values, size=num_samples)
        else:
            # For continuous features, perturb using normal distribution
            perturbations = np.random.normal(0, 1, size=num_samples)
            scaled_perturbations = perturbations * cont_std[col]
            perturbed_values = instance[dataset.columns.get_loc(col)] + scaled_perturbations

            # Clip the values to be within the range of the original dataset
            min_val = dataset[col].min()
            max_val = dataset[col].max()
            perturbed_samples[col] = np.clip(perturbed_values, min_val, max_val)

    return perturbed_samples.values.astype('int64')


def evaluate_local_model(explainer, instance, black_box_model, dataset, num_samples=1000):
    exp = explainer.explain_instance(instance, black_box_model.predict_proba, num_features=len(instance))
    available_labels = list(exp.local_exp.keys())
    if not available_labels:
        raise ValueError("No labels found in the explanation")
    label = available_labels[0]
    local_model = lambda x: exp.intercept[label] + np.dot(x, [exp.local_exp[label][i][1] for i in
                                                              range(len(exp.local_exp[label]))])
    perturbed_samples = generate_perturbed_samples(instance, dataset, num_samples)
    local_preds = np.array([local_model(x) > 0.5 for x in perturbed_samples])

    # Ensure perturbed_samples is a numpy array before passing to black_box_model
    if isinstance(perturbed_samples, pd.DataFrame):
        perturbed_samples = perturbed_samples.values

    black_box_preds = black_box_model.predict_proba(perturbed_samples)[:, label] > 0.5
    accuracy = np.mean(local_preds == black_box_preds)
    return accuracy, exp


def extract_data():
    data = pd.read_csv(filename)
    # Change categorical columns to category types
    data[categorical_column_names] = data[categorical_column_names].astype('category')
    cat_cols = data.select_dtypes(include=['category']).columns
    non_cat_cols = data.select_dtypes(exclude=['category']).columns
    cat_col_indices = [data.columns.get_loc(col) for col in cat_cols]
    non_cat_col_indices = [data.columns.get_loc(col) for col in non_cat_cols][:-1]
    # Save dict with key being the name of the cat column and the value being a dict,
    # with the key being the code of the category value, and the value being the category value itself
    category_mappings = {col: {v: k for k, v in enumerate(data[col].cat.categories)} for col in cat_cols}
    # Change the values in the data to be the category codes
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], random_state=random_state)
    x_train[categorical_column_names] = x_train[categorical_column_names].astype('category')
    x_test[categorical_column_names] = x_test[categorical_column_names].astype('category')
    return x_train, x_test, y_train, y_test, category_mappings, cat_col_indices, non_cat_col_indices


def train_model(x_train, x_test, y_train, y_test):
    if os.path.exists("model_lin.pkl"):
        with open('model_lin.pkl', 'rb') as f:
            return pickle.load(f)
    model = SVC(kernel='linear', probability=True)
    model.fit(x_train, y_train)
    with open('model_lin.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model accuracy score: " + str(accuracy_score(y_test, model.predict(x_test))))
    return model


def lime_lin(x_train, model, y_train, vec):
    # explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, mode="classification", training_labels=y_train, categorical_features=non_cat_col_indices)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        x_train.values,
        mode="classification",
        feature_names=x_train.columns.tolist(),
        categorical_features=[i for i, col in enumerate(x_train.columns) if col not in CONT_FEAT],
        kernel_width=0.75 * np.sqrt(x_train.shape[1]),
        verbose=True
    )
    # accuracy, exp = evaluate_local_model(explainer, vec, model, x_train)
    explanation = explainer.explain_instance(vec, model.predict_proba)
    local_linear_explanation = explanation.as_map()[1]
    print(local_linear_explanation)
    coefs = [0] * (max(index for index, _ in local_linear_explanation) + 1)
    for index, value in local_linear_explanation:
        coefs[index] = value
    intercept = explanation.intercept[1]
    return coefs, intercept


def extract_names_and_conditions(line):
    constraints = []
    pattern = r'([\w.-]+)\s*([<>=!]=?)\s*([\w."]+)'
    matches = re.findall(pattern, line.rstrip())
    for match in matches:
        lhs, op, rhs = match
        constraints.append((lhs, op, rhs))
    return constraints


def load_constraints(path):
    f = open(path, 'r')
    constraints_txt = []
    for line in f:
        constraint = extract_names_and_conditions(line.rstrip())
        if 't1.' in line:
            if any(op in line for op in ['>=', '<=', '< ', '> ']):
                rev_constraint = []
                for pred in constraint:
                    if 't0.' in pred[0]:
                        rev_constraint.append((pred[0].replace('t0.', 't1.'), pred[1], (pred[2].replace('t1.', 't0.'))))
                    else:
                        rev_constraint.append((pred[0].replace('t1.', 't0.'), pred[1], (pred[2].replace('t1.', 't0.'))))
                rev_constraint_fixed = []
                for pred in rev_constraint:
                    rev_constraint_fixed.append(
                        f'{pred[0].replace("t0.", "cf_row.").replace("t1.", "df.")} {pred[1]} {pred[2].replace("t0.", "cf_row.").replace("t1.", "df.")}')
                constraints_txt.append(rev_constraint_fixed)
        constraint_fixed = []
        for pred in constraint:
            constraint_fixed.append(
                f'{pred[0].replace("t0.", "cf_row.").replace("t1.", "df.")} {pred[1]} {pred[2].replace("t0.", "cf_row.").replace("t1.", "df.")}')
        constraints_txt.append(constraint_fixed)

    def cons_func(df, cf_row, cons_id, exclude=None):
        if exclude is None:
            exclude = []
        df_mask = '('
        for pred in constraints_txt[cons_id]:
            if (pred.split('.')[1].split(' ')[0] in exclude) and ('cf_row' in pred):
                continue
            df_mask += f'({pred}) & '
        if len(df_mask[:-1]) == 0:
            if len(exclude) == 0:
                return pd.Series([False] * len(df), index=df.index)
            else:
                return pd.Series([True] * len(df), index=df.index)
        res = eval(df_mask[:-3] + ')')
        if type(res) not in [np.bool_, bool]:
            return res
        if not res:
            return pd.Series([False] * len(df), index=df.index)
        else:
            return pd.Series([True] * len(df), index=df.index)

    dic_cols = {}
    unary_cons_lst = []
    unary_cons_lst_single = []
    bin_cons = []
    cons_feat = [[] for _ in range(len(constraints_txt))]
    for index, cons in enumerate(constraints_txt):
        unary = True
        for pred in cons:
            col = pred.split('.')[1].split(' ')[0]
            if index not in dic_cols.get(col, []):
                dic_cols[col] = dic_cols.get(col, []) + [index]
            cons_feat[index].append(col)
            if unary:
                if 'df.' in pred:
                    unary = False
        if unary:
            if len(cons) == 1:
                unary_cons_lst_single.append(index)
            else:
                unary_cons_lst.append(index)
        else:
            bin_cons.append(index)
    f.close()
    return constraints_txt, dic_cols, cons_func, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons


def solve_with_constraints(row, dataset, found_points, constraints, cons_function, cons_feat, coefs_dic, intercept,
                           unary_cons_lst, unary_cons_lst_single, category_mappings, bin_cons):
    viable_cols = [col for col in dataset.columns if col not in fixed_column_names]
    must_cons = set()
    violation_set = []
    for cons in range(len(constraints)):
        non_follow_cons = cons_function(dataset, row, cons)
        violation_set.append(dataset.loc[non_follow_cons])
        if len(non_follow_cons) == 0:
            continue
        count = non_follow_cons.sum()
        # if count > 0:
        # must_cons.add(cons)
    combs = list(itertools.combinations(viable_cols, len(viable_cols)))
    for comb in combs:
        if not all(any(feat in comb for feat in cons_feat[cons]) for cons in must_cons):
            continue
        s = Optimize()
        vars = {}
        for col in comb:
            if col not in categorical_column_names:
                var = Real(col)
                vars[col] = var
                s.add(var >= dataset[col].min(), var <= dataset[col].max())
            else:
                var = Int(col)
                vars[col] = var
                # s.add(var >= 0, var <= len(dataset[col].cat.categories) - 1)
                s.add(var >= 0, var <= max(dataset[col]))
        # distance = 0
        # med = np.median([(row[var] - dataset[var].median()) for var in vars if var not in categorical_column_names])
        # for var in vars:
        #     if var not in categorical_column_names:
        #         distance += Sum(Abs(vars[var] - row[var]) / med)
        #     if var in categorical_column_names:
        #         distance += Sum(vars[var] != row[var])
        distance = 0
        for var in vars:
            if var not in categorical_column_names:
                median = dataset[var].median()
                med = np.median([abs(ds - median) for ds in dataset[var]])
                distance += Sum(Abs(vars[var] - row[var]) / med)
            if var in categorical_column_names:
                distance += Sum(vars[var] != row[var])
        delta = 5
        total_dist = -delta * diversity(vars, found_points, x_train) + distance
        added_intercept = 0
        for coefs_key in coefs_dic:
            if coefs_key not in vars:
                added_intercept += coefs_dic[coefs_key] * row[coefs_key]
        s.add(Sum([coefs_dic[var] * vars[var] for var in vars]) + intercept + added_intercept >= 0.501)
        inner_product = intercept
        for key in coefs_dic:
            if key in row.index:  # Ensure the key exists in the Series
                inner_product += coefs_dic[key] * row[key]
        print('ip before: ' + str(inner_product))
        for un_cons in unary_cons_lst + unary_cons_lst_single:
            un_clause = []
            for pred in constraints[un_cons]:
                if (pred.split('.')[1].split(' ')[0] not in comb) and ('cf_row' in pred):
                    op = pred.split('.')[1].split(' ')[1]
                    col = pred.split('.')[1].split(' ')[0]
                    val = pred.split('.')[1].split(' ')[2]
                    if col not in categorical_column_names:
                        val = float(val)
                    else:
                        val = category_mappings[col][val[1:-1]]
                    if not ops.get(op)(row[col], val):
                        break
                    continue
                op = pred.split('.')[1].split(' ')[1]
                col = pred.split('.')[1].split(' ')[0]
                val = pred.split('.')[1].split(' ')[2]
                if col not in categorical_column_names:
                    val = float(val)
                else:
                    # val = dataset[col].cat.categories.get_loc(category_mappings[col][val[1:-1]])
                    val = category_mappings[col][val[1:-1]]
                if op == '==':
                    un_clause.append(vars[col] == val)
                elif op == '!=':
                    un_clause.append(vars[col] != val)
                elif op == '>':
                    un_clause.append(vars[col] > val)
                elif op == '>=':
                    un_clause.append(vars[col] >= val)
                elif op == '<':
                    un_clause.append(vars[col] < val)
                elif op == '<=':
                    un_clause.append(vars[col] <= val)
            # Comment for non Constraints
            else:
                s.add(Not(And(un_clause)))

        val_cons_dfs = {}
        for cons in bin_cons:
            val_cons_set = dataset[cons_function(dataset, row, cons, comb)]
            val_cons_dfs[cons] = val_cons_set
            if len(val_cons_dfs[cons]) == 0: continue
            clauses = [[] for _ in range(len(val_cons_dfs[cons]))]
            for pred in constraints[cons]:
                if (pred.split('.')[1].split(' ')[0] not in comb) and ('cf_row' in pred):
                    continue
                first_clause, op, second_clause = pred.split(' ')
                cf_pred = first_clause if 'cf_row' in first_clause else second_clause
                col = cf_pred.split('.')[1]
                if cf_pred == second_clause:
                    if op == '>':
                        op = '<'
                    elif op == '>=':
                        op = '<='
                    elif op == '<=':
                        op = '>='
                    elif op == '<':
                        op = '>'
                if col not in categorical_column_names:
                    it = val_cons_dfs[cons][col]
                else:
                    # it = val_cons_dfs[cons][col].cat.codes
                    it = val_cons_dfs[cons][col]
                for i, val in enumerate(it):
                    if op == '==':
                        clauses[i].append(vars[col] == val)
                    elif op == '!=':
                        clauses[i].append(vars[col] != val)
                    elif op == '>':
                        clauses[i].append(vars[col] > val)
                    elif op == '>=':
                        clauses[i].append(vars[col] >= val)
                    elif op == '<':
                        clauses[i].append(vars[col] < val)
                    elif op == '<=':
                        clauses[i].append(vars[col] <= val)
            clauses = set(tuple(lst) for lst in clauses)
            clauses = [list(tpl) for tpl in clauses]
            # Comment for non Constraints
            s.add([Not(And(clause)) for clause in clauses])
        opt = s.minimize(total_dist)
        s.set(timeout=10000)
        res = s.check().r
        s.lower(opt)
        if res != -1:
            m = s.model()
            row_val = row.copy()
            for col in comb:
                if col not in categorical_column_names:
                    value = m[vars[col]].as_decimal(10).replace('?', '')
                    row_val[col] = float(value)
                else:
                    # row_val[col] = dataset[col].cat.categories[m[vars[col]].as_long()]
                    row_val[col] = m[vars[col]].as_long()
            print('Success')
            print(row_val)
            return row_val
        print("Failure")


def dpp_style(cfs, vars, dataset):
    num_cfs = len(cfs)
    det_entries = torch.ones((num_cfs, num_cfs))
    for i in range(num_cfs):
        for j in range(num_cfs):
            det_entries[(i, j)] = 1.0 / (1.0 + compute_dist(cfs.iloc[i], vars, dataset))
            if i == j:
                det_entries[(i, j)] += 0.0001
    diversity_loss = torch.det(det_entries)
    return diversity_loss


def compute_dist(point, vars, dataset):
    distance = 0
    med = np.median([(point[var] - dataset[var].median()) for var in vars if var not in categorical_column_names])
    for var in vars:
        if var not in categorical_column_names:
            distance += Sum(Abs(vars[var] - point[var]) / med)
        if var in categorical_column_names:
            distance += Sum(vars[var] != point[var])
    return distance


if __name__ == "__main__":
    # x_train, x_test, y_train, y_test, category_mappings, cat_col_indices, non_cat_col_indices = extract_data()
    x_train, x_test, y_train, y_test, category_mappings, cat_col_indices, non_cat_col_indices = extract_data()

    # model_for_lime = RandomForestClassifier(n_estimators=100, random_state=42)
    # model_for_lime = SVC(kernel='linear',probability=True)
    # model_for_lime.fit(x_train, y_train)
    # with open('model2.pkl', 'wb') as f:
    #     pickle.dump(model_for_lime, f)
    # with open('data_ron2.pkl', 'rb') as f:
    #     data = pickle.load(f)
    with open('data_avia.pkl', 'rb') as f:
        data = pickle.load(f)
    x_train, x_test, y_train, y_test = data[6].drop(['label'], axis=1), data[7].drop(['label'], axis=1), data[6][
        'label'], data[7]['label']
    model = train_model(x_train, x_test, y_train, y_test)
    coefs_l, intercept_l = model.coef_[0], model.intercept_[0]
    # rows_to_run = x_test[y_test == 0][:num_of_rows_to_run]
    rows_to_run = data[3].drop('label', axis=1)
    # coefs_dic = {col: val for col, val in zip(x_train.columns, coefs)}
    constraints, dic_cols, cons_function, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons = load_constraints(
        constraints_name)
    returned_lst = [[] for _ in range(num_of_rows_to_run)]
    for i in range(num_of_rows_to_run):
        for attempt in range(num_of_cfs):
            # perturbed_samples = generate_perturbed_samples(rows_to_run.iloc[i], x_train)
            # model_limey = SVC(kernel='linear', probability=True)
            # y_limey = model_for_lime.predict(perturbed_samples)
            # model_limey.fit(np.concatenate([perturbed_samples[y_limey == 0][:len(perturbed_samples[y_limey == 1])],perturbed_samples[y_limey == 1]]),np.concatenate([y_limey[y_limey == 0][:len(y_limey[y_limey == 1])],y_limey[y_limey == 1]]))
            # coefs_ly,intercept_ly = model_limey.coef_[0], model_limey.intercept_[0]
            # coefs_l, intercept_l = lime_lin(x_train, model_for_lime, y_train, rows_to_run.iloc[i])
            # print(model_for_lime.coef_[0], model_for_lime.intercept_[0])
            # print(coefs_l, intercept_l)
            # coefs_l, intercept_l = data[1][i, :], data[2][i]
            coefs_dic_l = {col: val for col, val in zip(x_train.columns, coefs_l)}
            returned_lst[i].append(
                solve_with_constraints(rows_to_run.iloc[i], x_train, returned_lst[i], constraints, cons_function,
                                       cons_feat, coefs_dic_l, intercept_l, unary_cons_lst, unary_cons_lst_single,
                                       category_mappings, bin_cons))
            # returned_lst[i].append(solve_with_constraints(rows_to_run.iloc[i], x_train, returned_lst[i], constraints, cons_function, cons_feat, coefs_dic_ly, intercept_ly, unary_cons_lst, unary_cons_lst_single, category_mappings, bin_cons))
        # print("Row " + str(i) + ": " + str(model_for_lime.predict(returned_lst[i])))
        # print(model_for_lime.predict_proba(returned_lst[i]))
        # print(model_for_lime.predict_proba([rows_to_run.iloc[i]]))
        distances = [None] * len(returned_lst[i])
        for instance in range(len(returned_lst[i])):
            inner_product = intercept_l
            for key in coefs_dic_l:
                if key in returned_lst[i][instance].index:  # Ensure the key exists in the Series
                    inner_product += coefs_dic_l[key] * returned_lst[i][instance][key]
            print("ip after: " + str(inner_product))
            med = np.median([(returned_lst[i][instance][j] - x_train.iloc[:, j].median()) for j in non_cat_col_indices])
            # distances[instance] = np.sum([(abs(returned_lst[i][instance][j] - rows_to_run.iloc[instance, j])) / med for j in non_cat_col_indices])
            distances[instance] = np.sum(
                [(abs(returned_lst[i][instance][j] - rows_to_run.iloc[instance, j])) for j in non_cat_col_indices])
            distances[instance] += np.sum(
                [(returned_lst[i][instance][j] != rows_to_run.iloc[instance, j]) for j in cat_col_indices])
        print(distances)
        for instance in range(len(returned_lst[i])):
            for cons in range(len(constraints)):
                non_follow_cons = cons_function(x_train, returned_lst[i][instance], cons)
                if non_follow_cons.sum() != 0:
                    print("failed constraints " + str(constraints[cons]))
        # for instance in range(len(returned_lst[i])):
        #     for col in returned_lst[i][instance].items():
        #         if col[0] in categorical_column_names:
        #             if col[1] < 0 or col[1] > (len(x_train[col[0]].cat.categories) - 1):
        #                 print("failed column code constraint " + str(col))
        #         else:
        #             if col[1] < x_train[col[0]].min() or col[1] > x_train[col[0]].max():
        #                 print("failed column code constraint " + str(col))
    # with open('our_lime.pkl')
    pass
