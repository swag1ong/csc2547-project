"""
Calculate mean of min LUT-6 of first # epochs and last # epochs. By making comparisons, we can observe the learning results.
"""
import csv

min_old = []
min_new = []
file_num = 5  # the number of files taken into account
total_file_number = 50  # total amount of log files

# average min of first # of epochs
for i in range(file_num):

    fn = 'logs/log' + str(i + 1) + '.csv'
    min_lut = 999

    with open(fn) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # skip the titles
                line_count += 1

            elif line_count == 1:
                # record starting LUT
                lut_start = int(row[2])
                min_lut = int(row[2])
                line_count += 1
            else:
                # find and record the minimum
                if int(row[2]) < min_lut:
                    min_lut = int(row[2])
                    line_count += 1
    min_old.append(min_lut)

# calculate the mean
min_old_avg = sum(min_old) / len(min_old)

# average min of last # of epochs
for i in range(file_num):

    fn = 'logs/log' + str(total_file_number - i) + '.csv'
    min_lut = 999

    with open(fn) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # skip the titles
                line_count += 1
            else:
                # find and record the minimum
                if int(row[2]) < min_lut:
                    min_lut = int(row[2])
                    line_count += 1
    min_new.append(min_lut)

# calculate the mean
min_new_avg = sum(min_new) / len(min_new)

# make comparisons
print('The initial LUT-6 is:', lut_start)
print('Averge minimum LUT-6 of first', file_num, ' epoches is:', min_old_avg)
print('Averge minimum LUT-6 of last', file_num, ' epoches is:', min_new_avg)
print('Improvement is: ', min_old_avg - min_new_avg)
