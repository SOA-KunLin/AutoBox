from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import json
import math
import numpy

def create_app():
    app = Flask(__name__)
    #app.config.from_envvar('FLASK_APP_SETTINGS')
    return app

app = create_app()


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/upload_protein', methods=['POST'])
def upload_protein():
    file = request.files['file']
    filename = secure_filename(file.filename)
    
    # Is a pdb file
    if '.' not in filename or filename.rsplit('.', 1)[1] != 'pdb':
        return json.dumps({'message': 'fail', 'reason': 'not a pdb'})
    
    try:
        pdb_text = file.read().decode("utf-8")
    
        mol = load_pdb(pdb_text)
        new_xyz = gridbox_fitting(mol.xyz)
        mol.xyz = new_xyz
        
        [x, y, z] = [0, 0, 0]
        [rx, ry, rz] = numpy.ceil(get_global_gridbox_size(mol.xyz, margin=0.0)) # half of the side length and ceiling
        [deg_x, deg_y, deg_z] = [0, 0, 0]
    
        return json.dumps({
                   'message': 'success',
                   'pdb': mol.write_pdb(),
                   'x': x,
                   'y': y,
                   'z': z,
                   'rx': rx,
                   'ry': ry,
                   'rz': rz,
                   'deg_x': deg_x,
                   'deg_y': deg_y,
                   'deg_z': deg_z,
               })
    except:
        return json.dumps({'message': 'fail', 'reason': 'file cannot be processed'})


@app.route('/apply_gridbox', methods=['POST'])
def apply_gridbox():
    unit_deg = math.pi / 180
    
    pdb_text = request.form['pdb']
    x = int(request.form['x'])
    y = int(request.form['y'])
    z = int(request.form['z'])
    rx = float(request.form['rx'])
    ry = float(request.form['ry'])
    rz = float(request.form['rz'])
    deg_x = int(request.form['deg_x']) * unit_deg;
    deg_y = int(request.form['deg_y']) * unit_deg;
    deg_z = int(request.form['deg_z']) * unit_deg;
    
    mol = load_pdb(pdb_text)
    R = rotation_matrix_3d(deg_x, deg_y, deg_z).T
    mol.xyz = rotate_coords(mol.xyz, R)
    [[x, y, z]] = numpy.round(rotate_coords(numpy.asarray([[x, y, z]]), R))
    [deg_x, deg_y, deg_z] = [0, 0, 0]
    
    return json.dumps({
               'message': 'success',
               'pdb': mol.write_pdb(),
               'x': x,
               'y': y,
               'z': z,
               'rx': rx,
               'ry': ry,
               'rz': rz,
               'deg_x': deg_x,
               'deg_y': deg_y,
               'deg_z': deg_z,
           })
    

################################################################################
def load_pdb(pdb_text):
    pdb_text = pdb_text.split('\n')
    def gen_filter_ATOM(pdb_text):
        for line in pdb_text:
            if line.startswith('ATOM'):
                yield line
            if line.startswith('TER'):
                break
    ATOM_text = list(gen_filter_ATOM(pdb_text))
    
    def gen_ATOM(ATOM_text):
        for line in ATOM_text:
            yield ATOM(
                record = line[0:6],
                id = int(line[6:11]),
                name = line[11:16],
                resname = line[17:20],
                chain = line[21:22],
                resid = line[22:26],
                xyz = numpy.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                occ = float(line[54:60]),
                bval = float(line[60:66]),
                element = line[76:78],
            )
    atoms = list(gen_ATOM(ATOM_text))
    mol = Molecule(atoms)
    return mol


class ATOM(object):
    def __init__(self, record, id, name, resname, chain, resid, xyz, occ, bval, element):
        self._record = record
        self._id = id
        self._name = name
        self._resname = resname
        self._chain = chain
        self._resid = resid
        self._xyz = xyz
        self._occ = occ
        self._bval = bval
        self._element = element
        
    @property
    def record(self):
        return self._record
        
    @property
    def id(self):
        return self._id
        
    @property
    def name(self):
        return self._name
        
    @property
    def resname(self):
        return self._resname
        
    @property
    def chain(self):
        return self._chain
        
    @property
    def resid(self):
        return self._resid
        
    @property
    def xyz(self):
        return self._xyz
    
    
    @xyz.setter
    def xyz(self, xyz):
        self._xyz = xyz
    
        
    @property
    def occ(self):
        return self._occ
        
    @property
    def bval(self):
        return self._bval
        
    @property
    def element(self):
        return self._element
    
    
class Molecule(object):
    def __init__(self, ATOMS):
        self._ATOMS = ATOMS
        
    
    @property
    def xyz(self):
        return numpy.asarray([atom.xyz for atom in self._ATOMS])
    
    
    @xyz.setter
    def xyz(self, new_xyz):
        for xyz, atom in zip(new_xyz, self._ATOMS):
            atom.xyz = xyz 
    
        
    def write_pdb(self):
        def gen_line(ATOMS):
            for atom in ATOMS:
                line = ''.join([
                    atom.record,
                    str(atom.id).rjust(5, ' '),
                    atom.name,
                    ' ' * 1,
                    atom.resname,
                    ' ' * 1,
                    atom.chain,
                    atom.resid,
                    ' ' * 4,
                    '{0:8.3f}'.format(round(atom.xyz[0],3)),
                    '{0:8.3f}'.format(round(atom.xyz[1],3)),
                    '{0:8.3f}'.format(round(atom.xyz[2],3)),
                    '{0:6.2f}'.format(round(atom.occ,2)),
                    '{0:6.2f}'.format(round(atom.bval,2)),
                    ' ' * 10,
                    atom.element,
                    ' ' * 2,
                ])
                yield line
        return '\n'.join(list(gen_line(self._ATOMS)))
    
    
################################################################################
def gridbox_fitting(coords_2d):
    
    # Translate geometric center to grid box center
    gridboxcenter = numpy.asarray([0, 0, 0])
    geometrycenter = get_geometric_center(coords_2d)
    T = gridboxcenter - geometrycenter
    coords_translated = translate_coords(coords_2d, T)
    
    # Rotate first PC axis to align with grid box diagonal axis
    PC1 = get_first_PC_axis(coords_translated)
    gridboxDiagonalAxis = norm_v(numpy.asarray([1,1,1]))
    rotationAxis = get_rotation_axis(PC1, gridboxDiagonalAxis)
    rotationAngle = get_angle(PC1, gridboxDiagonalAxis)
    R = Rodrigues_rotation_matrix(rotationAxis, rotationAngle)
    coords_rotated = rotate_coords(coords_translated, R)
    
    # Translate the middle point to grid box center
    gridboxcenter = numpy.asarray([0, 0, 0])
    middle_point = (coords_rotated.max(axis=0) + coords_rotated.min(axis=0)) / 2
    T = gridboxcenter - middle_point
    coords_centered = translate_coords(coords_rotated, T)
    
    return coords_centered


def translate_coords(coords, T):
    # coords: 2d ndarray
    return coords + T


def get_geometric_center(coords):
    # coords: 2d ndarray
    return coords.mean(axis=0).squeeze()


def eig(X):
    # X is p x n matrix (p features, n points).
    
    centered_matrix = X - X.mean(axis=1)[:, numpy.newaxis]
    cov = numpy.dot(centered_matrix, centered_matrix.T)
    eigvals, eigvecs = numpy.linalg.eig(cov)
    
    idx = eigvals.argsort()
    return (eigvals[idx], eigvecs[:, idx])


def get_first_PC_axis(coords):
    # coords: 2d ndarray
    X = coords.T
    eigvals, PCs = eig(X)
    return PCs[:, -1]


def norm_v(V):
    # V is 1d ndarray.
    return V / numpy.linalg.norm(V)


def get_rotation_axis(from_v, to_v):
    # from_v, to_v: 1d ndarray
    return norm_v(numpy.cross(from_v, to_v))


def get_angle(v1, v2):
    # v1, v2 are 1d ndarray
    v1_n = norm_v(v1)
    v2_n = norm_v(v2)
    return numpy.arccos(numpy.clip(numpy.dot(v1_n, v2_n), -1.0, 1.0))


def Rodrigues_rotation_matrix(axis, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general
    axis by a specified angle.

    R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)
    (Rodrigues' rotation formula)

    Parameters:

        angle : float (a, angle)
        axis : 1d numpyarray (d, direction)
    """
    
    d = norm_v(axis)
    eye = numpy.eye(3, dtype=numpy.float64)
    ddt = numpy.outer(d, d)

    skew = numpy.array([[    0,  d[2], -d[1]],
                        [-d[2],     0,  d[0]],
                        [d[1] , -d[0],    0]], dtype=numpy.float64)

    mtx = ddt + numpy.cos(angle) * (eye - ddt) + numpy.sin(angle) * skew

    return mtx


def rotation_matrix_3d(theta_x, theta_y, theta_z):
    tx = theta_x
    ty = theta_y
    tz = theta_z
    
    sin = numpy.sin
    cos = numpy.cos
    
    Rx = numpy.array([[1, 0, 0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
    Ry = numpy.array([[cos(ty), 0, sin(ty)], [0, 1, 0], [-sin(ty), 0, cos(ty)]])
    Rz = numpy.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0, 0, 1]])
    
    return numpy.dot(Rx, numpy.dot(Ry, Rz))


def rotate_coords(coords, R):
    # coords: 2d ndarray
    return numpy.dot(R, coords.T).T


def get_global_gridbox_size(coords, margin=5.0):
    """
    get x, y, z side length of a gridbox that cover all the protein
    with margin of [margin] angstron.
    """
    # coords: 2d ndarray (nx3)
    return (coords.max(axis=0) - coords.min(axis=0)) + margin



if __name__ == '__main__':
    app.run(host='localhost', port=6465, debug=True, use_reloader=True)
