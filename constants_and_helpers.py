import numpy as np

gravitational_constant = 6.674*10**(-11)

g0 = 9.80665

def quat_multiply(q1, q2):
    w1, x1, y1, z1, w2, x2, y2, z2 = q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3]
    output_quat = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*x2])
    return output_quat

def quat_conjugate(q1):
    output_quat = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    return output_quat

def rotate_vector_by_quat(input_vector, input_quat):
    quat_vector = np.array([0.0, input_vector[0], input_vector[1], input_vector[2]])
    rotated_quat_vector = quat_multiply(input_quat, quat_multiply(quat_vector, quat_conjugate(input_quat)))
    return np.array([rotated_quat_vector[0], rotated_quat_vector[1], rotated_quat_vector[2]])