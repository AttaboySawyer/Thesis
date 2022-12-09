# # distance between the anchor and the positive
# pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
# print(pos_dist)
# # distance between the anchor and the negative
# neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)
# print(neg_dist)

# Save the dataset if results are good
# normal_opti_filder = "E:/Data/Optimized Sets/Normal"
# finding_opti_folder = "E:/Data/Optimized Sets/Findings"

# if(correct_preds/len(test_data) > 0.7):
#     for i in range(len(positive_files)):
#         image = cv2.imread(positive_files[i])
#         # print(positive_files[i])
#         filename = positive_files[i].replace('E:/Data/Ulcer_Training_And_Testing/Normal_Clean_Mucosa','')
#         # print(filename)
#         cv2.imwrite(normal_opti_filder+filename, image)
#     for i in range(len(negative_files)):
#         image = cv2.imread(negative_files[i])
#         # print(negative_files[i])
#         filename = negative_files[i].replace('E:/Data/Ulcer_Training_And_Testing/Ulcer_Confirmed','')
#         # print(filename)
#         cv2.imwrite(finding_opti_folder+filename, image)