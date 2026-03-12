import argparse
import json


def get_score(input_file):
    fail_num = 0
    task_id_list = []
    task_name_list = []
    total_num_dict = {}
    correct_num_dict = {}

    task_total_num_dict = {}
    task_correct_num_dict = {}

    with open(input_file, "r") as fp:
        all_lines = json.load(fp)
        for line in all_lines:
            task_name = line["task_name"]
            dataset_name = line["dataset_name"]
            if task_name is None:
                print("1.task_name is None")
                continue
            task_id = task_name + "_" + dataset_name
            if task_id not in task_id_list:
                task_id_list.append(task_id)
            if task_name not in task_name_list:
                task_name_list.append(task_name)
            total_num = total_num_dict.get(task_id, 0)
            correct_num = correct_num_dict.get(task_id, 0)
            task_total_num = task_total_num_dict.get(task_name, 0)
            task_correct_num = task_correct_num_dict.get(task_name, 0)
            predict = str(line["model_prediction"]).strip().replace("\n", "")
            # try:
            #     predict = int(predict) # from str to int
            #     predict = ["A", "B", "C", "D"][predict] # get letter version
            # except:
            #     print('Error in int to letter prediction!')
            #     raise ValueError()
            # if predict != 'None' and predict:
            #     if predict[0] == 'A' or predict[0] == 'B' or predict[0] == 'C' or predict[0] == 'D':
            #         gpt_predict = predict[0]
            #         if line['answer_gt'] == line['choice_a']:
            #             gt = 'A'
            #         elif line['answer_gt'] == line['choice_b']:
            #             gt = 'B'
            #         elif line['answer_gt'] == line.get('choice_c', None):
            #             gt = 'C'
            #         elif line['answer_gt'] == line.get('choice_d', None):
            #             gt = 'D'
            #         else:
            #             print('???? gt_answer is: ', end='')
            #             print(line['answer_gt'])
            #             exit(1)
            #     #This situation may occur when the answer given by gpt is "The answer is A."
            #     elif len(predict) > 1:
            #         if predict[-2] == 'A' or predict[-2] == 'B' or predict[-2] == 'C' or predict[-2] == 'D':
            #             gpt_predict = predict[-2]
            #             if line['answer_gt'] == line['choice_a']:
            #                 gt = 'A'
            #             elif line['answer_gt'] == line['choice_b']:
            #                 gt = 'B'
            #             elif line['answer_gt'] == line.get('choice_c', None):
            #                 gt = 'C'
            #             elif line['answer_gt'] == line.get('choice_d', None):
            #                 gt = 'D'
            #             else:
            #                 print('???? gt_answer is: ', end='')
            #                 print(line['answer_gt'])
            #                 exit(1)
            #         else:
            #             print(f'response is {predict}')
            #             fail_num += 1
            #             continue
            #     else:
            #         print(f'response is {predict}')
            #         fail_num += 1
            #         continue

            #     if gt == gpt_predict:
            #         total_num += 1
            #         correct_num += 1
            #     else:
            #         total_num += 1

            #     total_num_dict[task_id] = total_num
            #     correct_num_dict[task_id] = correct_num

            # else:
            #     print('2.Response is None.')
            #     fail_num += 1

            # it is safe to always choose the first option if the model fails
            if predict == "None":
                predict = str(line["choice_a"]).strip().replace("\n", "")

            answer_gt = str(line["answer_gt"]).strip().replace("\n", "")
            if predict == answer_gt:
                total_num += 1
                correct_num += 1
                task_total_num += 1
                task_correct_num += 1
            else:
                total_num += 1
                task_total_num += 1

            total_num_dict[task_id] = total_num
            correct_num_dict[task_id] = correct_num
            task_total_num_dict[task_name] = task_total_num
            task_correct_num_dict[task_name] = task_correct_num

    total_sum = 0
    total_answered = 0
    for task_id in task_id_list:
        total_num = total_num_dict[task_id]
        correct_num = correct_num_dict[task_id]
        acc = correct_num / total_num
        total_sum += total_num
        total_answered += correct_num
        print(f"{task_id}: Sum={total_num}, correct={correct_num}, acc={acc}")

    print(f"total_sum: {total_sum}")
    print(f"total_acc: {total_answered / total_sum}")
    print(f"fail_num: {fail_num}")

    category_acc = {}

    for task_name in task_name_list:
        total_num = task_total_num_dict[task_name]
        correct_num = task_correct_num_dict[task_name]
        acc = correct_num / total_num
        total_sum += total_num
        total_answered += correct_num
        category_acc[task_name] = acc * 100
        print(f"{task_name}: Sum={total_num}, correct={correct_num}, acc={acc}")

    return category_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a JSON file and calculate accuracy."
    )
    parser.add_argument("--input", type=str, help="Path to the input JSON file")
    args = parser.parse_args()

    get_score(args.input)
