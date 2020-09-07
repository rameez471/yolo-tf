import tensorflow as tf

from model.utils import broadcast_iou, xywh_to_x1x2y1y2

class Postprocessor(object):

    def __init__(self,iou_threshold,score_threshold,max_detection=100):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detection = max_detection

    def __call__(self,raw_yolo_output):

        boxes,objectness,class_probability = [],[],[]

        for i in raw_yolo_output:
            batch_size = tf.shape(i[0])[0]
            num_classes = tf.shape(i[2])[-1]
            boxes.append(tf.reshape(i[0],(batch_size,-1,4)))
            objectness.append(tf.reshape(i[1],(batch_size,-1,1)))
            class_probability.append(tf.reshape(i[2],(batch_size,-1,num_classes)))

        boxes = xywh_to_x1x2y1y2(tf.concat(boxes,axis=1))
        objectness = tf.concat(objectness,axis=1)
        class_probability = tf.concat(class_probability,axis=1)

        scores = objectness
        scores = tf.reshape(scores,(tf.shape(scores)[0],-1,tf.shape(scores)[-1]))

        final_boxes,final_scores, final_classes,valid_detections = self.batch_non_maximum_supression(
            boxes,scores,class_probability,self.iou_threshold,self.score_threshold,
            self.max_detection
        )

        return final_boxes, final_scores, final_classes, valid_detections

    @staticmethod
    def batch_non_maximum_supression(boxes,scores,classes,iou_threshold,score_threshold,max_detection):

        def single_batch_nms(candidate_boxes):
            candidate_boxes = tf.boolean_mask(
                candidate_boxes,candidate_boxes[...,4]>=score_threshold)
            outputs = tf.zeros((max_detection+1,tf.shape(candidate_boxes)[-1]))

            indices = []
            updates = []

            count = 0

            while tf.shape(candidate_boxes)[0] > 0 and count < max_detection:
                best_idx = tf.math.argmax(candidate_boxes[...,4],axis=0)
                best_box = candidate_boxes[best_idx]

                indices.append([count])
                updates.append(best_box)
                count += 1

                candidate_boxes = tf.concat([
                    candidate_boxes[0:best_idx],
                    candidate_boxes[best_idx+1:tf.shape(candidate_boxes)[0]]
                ],axis=0)

                iou = broadcast_iou(best_box[0:4],candidate_boxes[...,0:4])
                candidate_boxes = tf.boolean_mask(candidate_boxes,iou[0]<=iou_threshold)

            if count > 0:
                count_index = [[max_detection]]
                count_updates = [
                    tf.fill([tf.shape(candidate_boxes)[-1]],count)
                ]
                indices = tf.concat([indices,count_index],axis=0)
                updates = tf.concat([updates,count_updates],axis=0)
                outputs = tf.tensor_scatter_nd_update(outputs,indices,updates)

            return outputs

        combined_boxes = tf.concat([boxes,scores,classes],axis=2)
        result = tf.map_fn(single_batch_nms,combined_boxes)

        valid_counts = tf.expand_dims(
            tf.map_fn(lambda x: x[max_detection][0],result),axis=-1
        )
        final_result = tf.map_fn(lambda x: x[0:max_detection],result)
        nms_boxes,nms_scores,nms_classes = tf.split(
            final_result,[4,1,-1],axis=-1
        )

        return nms_boxes,nms_scores,nms_classes,tf.cast(
            valid_counts,tf.int32
        )