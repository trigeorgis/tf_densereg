from functools import lru_cache
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
from menpo.model import PCAModel
from menpo.shape import TriMesh
from scipy.io import loadmat


def load_tassos_lsfm_combined_model(path):
    m = loadmat(str(path))
    mean = TriMesh(m['mean'].reshape([-1, 3]), trilist=m['trilist'])
    return {
        'shape_model': PCAModel.init_from_components(
            m['components'].T,  m['eigenvalues'].ravel(),
            mean, 8888, True),
        'n_id_comps': int(m['n_trunc_ids'][0][0]),
        'n_exp_comps': int(m['n_trunc_expressions'][0][0])
    }


TEMPLATE_LMS_PATH = '/vol/atlas/databases/lsfm/template.ljson'

# load the maps between LSFM/Basel etc
@lru_cache()
def map_tddfa_to_basel():
    maps = mio.import_pickle(
        '/vol/atlas/databases/itwmm/mapping_mein3d_to_tddfa.pkl.gz')
    return maps['map_tddfa_to_basel']


def fw_to_fw_cropped():
    return mio.import_pickle('/vol/atlas/databases/itwmm/3ddfa_to_trimmed_no_neck_mask.pkl.gz')


@lru_cache()
def template_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/mein3d_fw_correspond_mean.pkl.gz')


# Remap basel landmarks to fw landmarks by expressing as fw indices
@lru_cache()
def fw_index_for_lms():
    basel_model, landmarks = load_basel_shape()
    basel_mean = basel_model.mean()
    basel_index = np.argmin(basel_mean.distance_to(landmarks), axis=0)

    m = np.ones(basel_mean.n_points) * -1
    m[basel_index] = np.arange(68)

    poses = np.where((m[map_tddfa_to_basel()] >= 0))[0]

    new_ids = m[map_tddfa_to_basel()][poses]
    return poses[np.argsort(new_ids)]


def template_fw_cropped():
    return template_fw().from_mask(fw_to_fw_cropped())


# Remappings between BFM [] - Face warehouse [fw] - Face warehouse cropped [fwc]
def map_basel_shape_model_to_fw(shape_model):
    shape_model = shape_model.copy()
    c  = shape_model._components.reshape([shape_model._components.shape[0], -1, 3])
    shape_model._components = c[:, map_tddfa_to_basel()].reshape([shape_model._components.shape[0], -1])
    shape_model._mean = shape_model._mean.reshape([-1, 3])[map_tddfa_to_basel()].ravel()
    shape_model.template_instance = template_fw().from_vector(shape_model._mean)
    return shape_model


def map_basel_shape_model_to_fwc(shape_model):
    shape_model = shape_model.copy()
    c  = shape_model._components.reshape([shape_model._components.shape[0], -1, 3])
    shape_model._components = c[:, map_tddfa_to_basel()][:, fw_to_fw_cropped()].reshape([shape_model._components.shape[0], -1])
    shape_model._mean = shape_model._mean.reshape([-1, 3])[map_tddfa_to_basel()][fw_to_fw_cropped()].ravel()
    shape_model.template_instance = template_fw_cropped().from_vector(shape_model._mean)
    return shape_model


def map_basel_texture_model_to_fw(texture_model):
    texture_model = texture_model.copy()
    c  = texture_model._components.reshape([texture_model._components.shape[0], -1, 3])
    texture_model._components = c[:, map_tddfa_to_basel()].reshape([texture_model._components.shape[0], -1])
    texture_model._mean = texture_model._mean.reshape([-1, 3])[map_tddfa_to_basel()].ravel()
    return texture_model


def map_basel_texture_model_to_fwc(texture_model):
    texture_model = texture_model.copy()
    c  = texture_model._components.reshape([texture_model._components.shape[0], -1, 3])
    texture_model._components = c[:, map_tddfa_to_basel()][:, fw_to_fw_cropped()].reshape([texture_model._components.shape[0], -1])
    texture_model._mean = texture_model._mean.reshape([-1, 3])[map_tddfa_to_basel()][fw_to_fw_cropped()].ravel()
    return texture_model


def load_basel_shape():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/shape_PCAModel.pkl', encoding='latin1')
    landmarks = m3io.import_landmark_file(TEMPLATE_LMS_PATH)
    return shape_model, landmarks


def load_basel_texture():
    return mio.import_pickle('/vol/atlas/databases/lsfm/texture_PCAModel.pkl',
                             encoding='latin1')


def load_basel_shape_fw():
    shape_model, landmarks = load_basel_shape()
    return map_basel_shape_model_to_fw(shape_model), landmarks


def load_basel_shape_fwc():
    shape_model, landmarks = load_basel_shape()
    return map_basel_shape_model_to_fwc(shape_model), landmarks


def load_basel_texture_fw():
    return map_basel_texture_model_to_fw(load_basel_texture())


def load_basel_texture_fwc():
    return map_basel_texture_model_to_fwc(load_basel_texture())


def load_lsfm_shape_fwc():
    tr = mio.import_pickle('/vol/atlas/databases/lsfm/corrective_translation.pkl')
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/lsfm_shape_model_fw_cropped.pkl')
    landmarks = tr.apply(m3io.import_landmark_file(TEMPLATE_LMS_PATH))
    return shape_model, landmarks


def load_lsfm_texture_fwc():
    return mio.import_pickle('/vol/atlas/databases/lsfm/colour_pca_model_fw_cropped.pkl')


def load_lsfm_combined_fw():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/lsfm_combined_model_fw.pkl')
    landmarks = m3io.import_landmark_file(TEMPLATE_LMS_PATH).from_vector(shape_model.mean().points[fw_index_for_lms()])
    return shape_model, landmarks


def load_basel_combined_fw():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/basel_combined_model_fw.pkl')
    landmarks = m3io.import_landmark_file(TEMPLATE_LMS_PATH).from_vector(shape_model.mean().points[fw_index_for_lms()])
    return shape_model, landmarks


def load_itwmm_texture_rgb_fwc():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_cropped/rgb/rgb_per_vertex_fw_cropped_texture_model.pkl')


def load_itwmm_texture_fast_dsift_fwc():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_cropped/fast_dsift/pca_model.pkl')


def load_itwmm_texture_fast_dsift_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw/fast_dsift.pkl')


def load_itwmm_texture_rgb_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw/rgb.pkl')


def load_itwmm_texture_no_mask_fast_dsift_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_no_mask/fast_dsift.pkl')


def load_itwmm_texture_no_mask_rgb_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_no_mask/rgb.pkl')


def load_fw_mean_id_expression_fw():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/expression_model_id_mean.pkl')
    landmarks = m3io.import_landmark_file(TEMPLATE_LMS_PATH).lms.from_vector(shape_model.mean().points[fw_index_for_lms()])
    return shape_model, landmarks


def load_fw_expression_fwc():
    expression_model = mio.import_pickle('./identity_texture_emotion.pkl')['expression']
    expression_model._components /= 100000
    expression_model._mean /= 100000
    tr = mio.import_pickle('/vol/atlas/databases/lsfm/corrective_translation.pkl')
    expression_model._components = tr.apply(expression_model._components.reshape(29, -1, 3)).reshape(29, -1)
    expression_model._mean = tr.apply(expression_model._mean.reshape(-1, 3)).reshape(-1)
    expression_model.n_active_components = 5
    return expression_model


def load_basel_concatenated_fw():
    x = mio.import_pickle(
        '/vol/atlas/homes/jab08/tpami_computational_face/concatenated_model.pkl')
    shape_model = x['combined_model']
    landmarks = m3io.import_landmark_file(TEMPLATE_LMS_PATH).from_vector(shape_model.mean().points[fw_index_for_lms()])
    return shape_model, landmarks, x['id_indices'], x['exp_indices']


def load_lsfm_tassos_concatenated_fw():
    x = load_tassos_lsfm_combined_model(
        '/vol/atlas/homes/aroussos/results/fit3Dto2D/model/ver2016-12-12_LSFMfrmt_maxNpcInf/all_all_all.mat')
    shape_model = x['shape_model']
    landmarks = m3io.import_landmark_file(TEMPLATE_LMS_PATH).from_vector(shape_model.mean().points[fw_index_for_lms()])
    id_indices = np.arange(x['n_id_comps'])
    exp_indices = np.arange(x['n_id_comps'], shape_model.n_components)
    return shape_model, landmarks, id_indices, exp_indices
