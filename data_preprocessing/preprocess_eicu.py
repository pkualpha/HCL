"""
This code is adapted from process steps on eICU of previous works (cited)
https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
"""

import csv
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from global_settings import dataset_path, raw_path
from scipy.sparse import csr_matrix
from sklearn import model_selection
from utils import load_pkl, write_pkl

tf.compat.v1.enable_eager_execution()


class EncounterInfo(object):
    def __init__(self, patient_id, encounter_id, encounter_timestamp, expired, readmission):
        self.patient_id = patient_id
        # patientHealthSystemStayID: surrogate key for the patient health system stay
        self.encounter_id = encounter_id
        # patientUnitStayID: surrogate key for ICU Stay
        self.encounter_timestamp = encounter_timestamp
        # hospitalAdmitOffset: number of minutes from unit admit time that the patient was admitted to the hospital
        self.expired = expired
        self.readmission = readmission
        # whether readmission to ICU
        self.dx_ids = []
        # From admissiondx: admitdxpath, diagnosis: diagnosisstring
        self.rx_ids = []
        self.labs = {}
        self.physicals = []
        self.treatments = []
        # From treatment: treatmentString


def process_patient(infile, encounter_dict, hour_threshold=24, verbose=False):
    inff = open(infile, "r")
    count = 0
    patient_dict = {}
    for line in csv.DictReader(inff):
        if verbose and count % 10000 == 0:
            sys.stdout.write("process_patient %d\r" % count)
            sys.stdout.flush()

        patient_id = line["patienthealthsystemstayid"]
        encounter_id = line["patientunitstayid"]
        encounter_timestamp = -int(line["hospitaladmitoffset"])
        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        patient_dict[patient_id].append((encounter_timestamp, encounter_id))
    inff.close()
    print("")

    patient_dict_sorted = {}
    for patient_id, time_enc_tuples in patient_dict.items():
        patient_dict_sorted[patient_id] = sorted(time_enc_tuples)

    enc_readmission_dict = {}
    for patient_id, time_enc_tuples in patient_dict_sorted.items():
        for time_enc_tuple in time_enc_tuples[:-1]:
            enc_id = time_enc_tuple[1]
            enc_readmission_dict[enc_id] = True
        last_enc_id = time_enc_tuples[-1][1]
        enc_readmission_dict[last_enc_id] = False

    inff = open(infile, "r")
    count = 0
    for line in csv.DictReader(inff):
        if verbose and count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        patient_id = line["patienthealthsystemstayid"]
        encounter_id = line["patientunitstayid"]
        encounter_timestamp = -int(line["hospitaladmitoffset"])
        discharge_status = line["unitdischargestatus"]
        duration_minute = float(line["unitdischargeoffset"])
        readmission = enc_readmission_dict[encounter_id]
        expired = True if discharge_status == "Expired" else False

        if duration_minute > 60.0 * hour_threshold:
            continue

        ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired, readmission)
        if encounter_id in encounter_dict:
            print("Duplicate encounter ID!!")
            sys.exit(0)
        encounter_dict[encounter_id] = ei
        count += 1

    inff.close()
    print("")

    return encounter_dict


def process_admission_dx(infile, encounter_dict, verbose=False):
    inff = open(infile, "r")
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if verbose and count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        encounter_id = line["patientunitstayid"]
        dx_id = line["admitdxpath"].lower()

        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].dx_ids.append(dx_id)
        count += 1
    inff.close()
    print("")
    print("Admission Diagnosis without Encounter ID: %d" % missing_eid)

    return encounter_dict


def process_diagnosis(infile, encounter_dict, verbose=False):
    inff = open(infile, "r")
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if verbose and count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        encounter_id = line["patientunitstayid"]
        dx_id = line["diagnosisstring"].lower()

        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].dx_ids.append(dx_id)
        count += 1
    inff.close()
    print("")
    print("Diagnosis without Encounter ID: %d" % missing_eid)

    return encounter_dict


def process_treatment(infile, encounter_dict, verbose=False):
    inff = open(infile, "r")
    count = 0
    missing_eid = 0

    for line in csv.DictReader(inff):
        if verbose and count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()
        encounter_id = line["patientunitstayid"]
        treatment_id = line["treatmentstring"].lower()
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].treatments.append(treatment_id)
        count += 1
    inff.close()
    print("")
    print("Treatment without Encounter ID: %d" % missing_eid)
    print("Accepted treatments: %d" % count)

    return encounter_dict


def build_seqex(enc_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50):
    key_list = []
    seqex_list = []
    dx_str2int = {}
    treat_str2int = {}
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_treatments = 0
    num_unique_dx_ids = 0
    num_unique_treatments = 0
    min_dx_cut = 0
    min_treatment_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    num_readmission = 0
    num_expired = 0

    for _, enc in enc_dict.items():
        if skip_duplicate:
            if len(enc.dx_ids) > len(set(enc.dx_ids)) or len(enc.treatments) > len(set(enc.treatments)):
                num_duplicate += 1
                continue

        if len(set(enc.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue

        if len(set(enc.treatments)) < min_num_codes:
            min_treatment_cut += 1
            continue

        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue

        if len(set(enc.treatments)) > max_num_codes:
            max_treatment_cut += 1
            continue

        count += 1
        num_dx_ids += len(enc.dx_ids)
        num_treatments += len(enc.treatments)
        num_unique_dx_ids += len(set(enc.dx_ids))
        num_unique_treatments += len(set(enc.treatments))

        for dx_id in enc.dx_ids:
            if dx_id not in dx_str2int:
                dx_str2int[dx_id] = len(dx_str2int)

        for treat_id in enc.treatments:
            if treat_id not in treat_str2int:
                treat_str2int[treat_id] = len(treat_str2int)

        seqex = tf.train.SequenceExample()
        seqex.context.feature["patientId"].bytes_list.value.append(
            bytes(enc.patient_id + ":" + enc.encounter_id, "utf-8")
        )

        if enc.expired:
            seqex.context.feature["label.expired"].int64_list.value.append(1)
            num_expired += 1
        else:
            seqex.context.feature["label.expired"].int64_list.value.append(0)

        if enc.readmission:
            seqex.context.feature["label.readmission"].int64_list.value.append(1)
            num_readmission += 1
        else:
            seqex.context.feature["label.readmission"].int64_list.value.append(0)

        # if enc.readmission:
        #     seqex.context.feature['label'].int64_list.value.append(1)
        #     num_readmission += 1
        # else:
        #     seqex.context.feature['label'].int64_list.value.append(0)

        dx_ids = seqex.feature_lists.feature_list["dx_ids"]
        dx_ids.feature.add().bytes_list.value.extend(list([bytes(s, "utf-8") for s in set(enc.dx_ids)]))

        dx_int_list = [dx_str2int[item] for item in list(set(enc.dx_ids))]
        dx_ints = seqex.feature_lists.feature_list["dx_ints"]
        dx_ints.feature.add().int64_list.value.extend(dx_int_list)

        proc_ids = seqex.feature_lists.feature_list["proc_ids"]
        proc_ids.feature.add().bytes_list.value.extend(list([bytes(s, "utf-8") for s in set(enc.treatments)]))

        proc_int_list = [treat_str2int[item] for item in list(set(enc.treatments))]
        proc_ints = seqex.feature_lists.feature_list["proc_ints"]
        proc_ints.feature.add().int64_list.value.extend(proc_int_list)

        seqex_list.append(seqex)
        key = seqex.context.feature["patientId"].bytes_list.value[0]
        key_list.append(key)

    print("Filtered encounters due to duplicate codes: %d" % num_duplicate)
    print("Filtered encounters due to thresholding: %d" % num_cut)
    print("Average num_dx_ids: %f" % (num_dx_ids / count))
    print("Average num_treatments: %f" % (num_treatments / count))
    print("Average num_unique_dx_ids: %f" % (num_unique_dx_ids / count))
    print("Average num_unique_treatments: %f" % (num_unique_treatments / count))
    print("Min dx cut: %d" % min_dx_cut)
    print("Min treatment cut: %d" % min_treatment_cut)
    print("Max dx cut: %d" % max_dx_cut)
    print("Max treatment cut: %d" % max_treatment_cut)
    print("Number of readmission: %d" % num_readmission)
    print("Number of expired: %d" % num_expired)

    return key_list, seqex_list, dx_str2int, treat_str2int


def select_train_valid_test(key_list, random_seed=0):
    train_id, val_id = model_selection.train_test_split(key_list, test_size=0.3, random_state=random_seed)
    test_id, val_id = model_selection.train_test_split(val_id, test_size=0.5, random_state=random_seed)
    return train_id, val_id, test_id


def get_partitions(seqex_list, id_set=None):
    total_visit = 0
    new_seqex_list = []
    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write("Visit count: %d\r" % total_visit)
            sys.stdout.flush()
        key = seqex.context.feature["patientId"].bytes_list.value[0]
        if id_set is not None and key not in id_set:
            total_visit += 1
            continue
        new_seqex_list.append(seqex)
    return new_seqex_list


def parser_fn(serialized_example):
    context_features_config = {
        "patientId": tf.VarLenFeature(tf.string),
        "label.expired": tf.FixedLenFeature([1], tf.int64),
        "label.readmission": tf.FixedLenFeature([1], tf.int64),
    }
    sequence_features_config = {
        "dx_ints": tf.VarLenFeature(tf.int64),
        "proc_ints": tf.VarLenFeature(tf.int64),
    }
    (batch_context, batch_sequence) = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=context_features_config,
        sequence_features=sequence_features_config,
    )
    expired_labels = tf.squeeze(tf.cast(batch_context["label.expired"], tf.float32))
    readmission_labels = tf.squeeze(tf.cast(batch_context["label.readmission"], tf.float32))
    return batch_sequence, expired_labels, readmission_labels


def tf2csr(output_path, maps, verbose=False):
    num_epochs = 1
    buffer_size = 32
    dataset = tf.data.TFRecordDataset(output_path + "/all.tfrecord")
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parser_fn, num_parallel_calls=4)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(16)
    count = 0
    np_data = []
    expired_label = []
    readmission_label = []
    for data in dataset:
        count += 1
        np_datum = np.zeros(sum([len(m) for m in maps]))
        dx_pos = tf.sparse.to_dense(data[0]["dx_ints"]).numpy().ravel()
        proc_pos = tf.sparse.to_dense(data[0]["proc_ints"]).numpy().ravel() + sum([len(m) for m in maps[:1]])
        np_datum[dx_pos] = 1
        np_datum[proc_pos] = 1
        np_data.append(np_datum)
        expired_label.append(data[1].numpy()[0])
        readmission_label.append(data[2].numpy()[0])
        if verbose:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

    pickle.dump(
        (
            csr_matrix(np.array(np_data)),
            np.array(expired_label),
            np.array(readmission_label),
        ),
        open(output_path + "/_csr.pkl", "wb"),
    )


def main():
    """
    Set <input_path> to where the raw eICU CSV files are located.
    Set <output_path> to where you want the output files to be.
    """
    input_path = os.path.join(raw_path, "eicu")
    output_path = os.path.join(dataset_path, "eicu")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    patient_file = input_path + "/patient.csv"
    admission_dx_file = input_path + "/admissionDx.csv"
    diagnosis_file = input_path + "/diagnosis.csv"
    treatment_file = input_path + "/treatment.csv"

    encounter_dict = {}
    print("Processing patient.csv")
    encounter_dict = process_patient(patient_file, encounter_dict, hour_threshold=24)
    print(len(encounter_dict))
    print("Processing admission diagnosis.csv")
    encounter_dict = process_admission_dx(admission_dx_file, encounter_dict)
    print("Processing diagnosis.csv")
    encounter_dict = process_diagnosis(diagnosis_file, encounter_dict)
    print("Processing treatment.csv")
    encounter_dict = process_treatment(treatment_file, encounter_dict)

    key_list, seqex_list, dx_map, proc_map = build_seqex(
        encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50
    )

    pickle.dump(dx_map, open(output_path + "/dx_map.p", "wb"), -1)
    pickle.dump(proc_map, open(output_path + "/proc_map.p", "wb"), -1)

    with tf.io.TFRecordWriter(output_path + "/all.tfrecord") as writer:
        for seqex in seqex_list:
            writer.write(seqex.SerializeToString())

    tf2csr(output_path, [dx_map, proc_map])

    x, dead_label, read_label = load_pkl(output_path + "/_csr.pkl")

    for i in range(10):
        seed = i * 7
        path = output_path + f"/fold_{i}"
        if not os.path.exists(path):
            os.mkdir(path)

        (X_train, X_test, dead_train, dead_test, read_train, read_test,) = model_selection.train_test_split(
            x, dead_label, read_label, test_size=0.3, stratify=dead_label, random_state=seed
        )

        (X_val, X_test, dead_val, dead_test, read_val, read_test,) = model_selection.train_test_split(
            X_test, dead_test, read_test, test_size=0.5, stratify=dead_test, random_state=seed
        )

        write_pkl(
            (X_train, dead_train, read_train),
            os.path.join(path, "train_csr.pkl"),
        )
        write_pkl((X_val, dead_val, read_val), os.path.join(path, "validation_csr.pkl"))
        write_pkl((X_test, dead_test, read_test), os.path.join(path, "test_csr.pkl"))

    # leave dataset splitting in dataloader
    # key_train, key_valid, key_test = select_train_valid_test(key_list)

    # train_seqex = get_partitions(seqex_list, set(key_train))
    # validation_seqex = get_partitions(seqex_list, set(key_valid))
    # test_seqex = get_partitions(seqex_list, set(key_test))

    # print("Split done.")

    # with tf.io.TFRecordWriter(output_path + '/train.tfrecord') as writer:
    #     for seqex in train_seqex:
    #         writer.write(seqex.SerializeToString())

    # with tf.io.TFRecordWriter(output_path + '/validation.tfrecord') as writer:
    #     for seqex in validation_seqex:
    #         writer.write(seqex.SerializeToString())

    # with tf.io.TFRecordWriter(output_path + '/test.tfrecord') as writer:
    #     for seqex in test_seqex:
    #         writer.write(seqex.SerializeToString())

    # for partition in ['train', 'validation', 'test']:
    #     tf2csr(output_path, partition, [dx_map, proc_map])
    # print('done')


def get_smaller_data():
    output_path = os.path.join(dataset_path, "eicu")

    x, dead_label, read_label = load_pkl(output_path + "/_csr.pkl")

    proc_map = load_pkl(output_path + "/proc_map.p")
    dx_map = load_pkl(output_path + "/dx_map.p")
    for k, v in proc_map.items():
        proc_map[k] = v + len(dx_map)
    proc_map_inv = {}
    for k, v in proc_map.items():
        proc_map_inv[v] = k
    dx_map_inv = {}
    for k, v in dx_map.items():
        dx_map_inv[v] = k

    f = np.array(x.sum(axis=0), dtype=int).squeeze()
    c = np.argwhere(f > 10).squeeze()

    xs = x[:, c]
    print(xs.shape)

    write_pkl((xs, dead_label, read_label), output_path + "/small_csr.pkl")

    dx_list = []
    for i in c[c < 3249]:
        dx_list.append(dx_map_inv[i])

    proc_list = []
    for i in c[c > 3248]:
        proc_list.append(proc_map_inv[i])

    small_dx_map = {}
    for i, v in enumerate(dx_list):
        small_dx_map[v] = i

    small_proc_map = {}
    for i, v in enumerate(proc_list):
        small_proc_map[v] = i

    write_pkl(small_dx_map, output_path + "/small_dx_map.pkl")
    write_pkl(small_proc_map, output_path + "/small_proc_map.pkl")

    x = xs
    for i in range(10):
        seed = i * 7
        path = output_path + f"/fold_{i}"
        if not os.path.exists(path):
            os.mkdir(path)

        (X_train, X_test, dead_train, dead_test, read_train, read_test,) = model_selection.train_test_split(
            x, dead_label, read_label, test_size=0.3, stratify=dead_label, random_state=seed
        )

        (X_val, X_test, dead_val, dead_test, read_val, read_test,) = model_selection.train_test_split(
            X_test, dead_test, read_test, test_size=0.5, stratify=dead_test, random_state=seed
        )

        write_pkl(
            (X_train, dead_train, read_train),
            os.path.join(path, "small_train_csr.pkl"),
        )
        write_pkl(
            (X_val, dead_val, read_val),
            os.path.join(path, "small_validation_csr.pkl"),
        )
        write_pkl(
            (X_test, dead_test, read_test),
            os.path.join(path, "small_test_csr.pkl"),
        )


if __name__ == "__main__":
    main()
    get_smaller_data()
