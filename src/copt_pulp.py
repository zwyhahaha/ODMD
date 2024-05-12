#
# This is the PuLP plugin of Cardinal Optimizer, all rights reserved
#

import os
import sys
import ctypes
import subprocess
import warnings

from uuid import uuid4
from pulp import sparse, ctypesArrayFill, PulpSolverError

from pulp import LpSolver, LpSolver_CMD
from pulp import LpStatusNotSolved, LpStatusOptimal, LpStatusInfeasible, \
                 LpStatusUnbounded, LpStatusUndefined
from pulp import LpContinuous, LpBinary, LpInteger
from pulp import LpConstraintEQ, LpConstraintLE, LpConstraintGE
from pulp import LpMinimize, LpMaximize


# COPT string convention
if sys.version_info >= (3, 0):
  coptstr = lambda x: bytes(x, 'utf-8')
else:
  coptstr = lambda x: x

byref = ctypes.byref


class COPT_CMD(LpSolver_CMD):
  """
  The COPT command-line solver
  """
  def __init__(self, path=None, keepFiles=0, mip=True, msg=True, mip_start=False, logfile=None, **params):
    """
    Initialize command-line solver
    """
    LpSolver_CMD.__init__(self, path, keepFiles, mip, msg, [])

    self.mipstart = mip_start
    self.logfile = logfile
    self.solverparams = params

  def defaultPath(self):
    """
    The default path of 'copt_cmd'
    """
    return self.executableExtension("copt_cmd")

  def available(self):
    """
    True if 'copt_cmd' is available
    """
    return self.executable(self.path)

  def actualSolve(self, lp):
    """
    Solve a well formulated LP problem

    This function borrowed implementation of CPLEX_CMD.actualSolve and
    GUROBI_CMD.actualSolve, with some modifications.
    """
    if not self.available():
      raise PulpSolverError("COPT_PULP: Failed to execute '{}'".format(self.path))

    if not self.keepFiles:
      uuid = uuid4().hex
      tmpLp  = os.path.join(self.tmpDir, "{}-pulp.lp".format(uuid))
      tmpSol = os.path.join(self.tmpDir, "{}-pulp.sol".format(uuid))
      tmpMst = os.path.join(self.tmpDir, "{}-pulp.mst".format(uuid))
    else:
      # Replace space with underscore to make filepath better
      tmpName = lp.name
      tmpName = tmpName.replace(" ", "_")

      tmpLp  = tmpName + "-pulp.lp"
      tmpSol = tmpName + "-pulp.sol"
      tmpMst = tmpName + "-pulp.mst"

    lpvars = lp.writeLP(tmpLp, writeSOS=1)

    # Generate solving commands
    solvecmds  = self.path
    solvecmds += " -c "
    solvecmds += "\"read " + tmpLp + ";"

    if lp.isMIP() and self.mipstart:
      self.writemst(tmpMst, lpvars)
      solvecmds += "read " + tmpMst + ";"

    if self.logfile is not None:
      solvecmds += "set logfile {};".format(self.logfile)

    if self.solverparams is not None:
      for parname, parval in self.solverparams.items():
        solvecmds += "set {0} {1};".format(parname, parval)

    if lp.isMIP() and not self.mip:
      solvecmds += "optimizelp;"
    else:
      solvecmds += "optimize;"

    solvecmds += "write " + tmpSol + ";"
    solvecmds += "exit\""

    try:
      os.remove(tmpSol)
    except:
      pass

    if self.msg:
      msgpipe = None
    else:
      msgpipe = open(os.devnull, 'w')

    rc = subprocess.call(solvecmds, shell=True, stdout=msgpipe, stderr=msgpipe)

    if msgpipe is not None:
      msgpipe.close()

    # Get and analyze result
    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to execute '{}'".format(self.path))

    if not os.path.exists(tmpSol):
      status = LpStatusNotSolved
    else:
      status, values = self.readsol(tmpSol)

    if not self.keepFiles:
      for oldfile in [tmpLp, tmpSol, tmpMst]:
        try:
          os.remove(oldfile)
        except:
          pass

    if status == LpStatusOptimal:
      lp.assignVarsVals(values)

    # lp.assignStatus(status)
    lp.status = status

    return status

  def readsol(self, filename):
    """
    Read COPT solution file
    """
    with open(filename) as solfile:
      try:
        next(solfile)
      except StopIteration:
        warnings.warn("COPT_PULP: No solution was returned")
        return LpStatusNotSolved, {}

      #TODO: No information about status, assumed to be optimal
      status = LpStatusOptimal

      values = {}
      for line in solfile:
        if line[0] != '#':
          varname, varval = line.split()
          values[varname] = float(varval)
    return status, values

  def writemst(self, filename, lpvars):
    """
    Write COPT MIP start file
    """
    mstvals = [(v.name, v.value()) for v in lpvars if v.value() is not None]
    mstline = []
    for varname, varval in mstvals:
      mstline.append('{0} {1}'.format(varname, varval))

    with open(filename, 'w') as mstfile:
      mstfile.write('\n'.join(mstline))
    return True


def COPT_DLL_loadlib():
  """
  Load COPT shared library in all supported platforms
  """
  from glob import glob

  libfile = None
  libpath = None
  libhome = os.getenv("COPT_HOME")

  if sys.platform == 'win32':
    libfile = glob(os.path.join(libhome, "bin", "copt.dll"))
  elif sys.platform == 'linux':
    libfile = glob(os.path.join(libhome, "lib", "libcopt.so"))
  elif sys.platform == 'darwin':
    libfile = glob(os.path.join(libhome, "lib", "libcopt.dylib"))
  else:
    raise PulpSolverError("COPT_PULP: Unsupported operating system")

  # Find desired library in given search path
  if libfile:
    libpath = libfile[0]

  if libpath is None:
    raise PulpSolverError("COPT_PULP: Failed to locate solver library, "
                          "please refer to COPT manual for installation guide")
  else:
    if sys.platform == 'win32':
      coptlib = ctypes.windll.LoadLibrary(libpath)
    else:
      coptlib = ctypes.cdll.LoadLibrary(libpath)

  return coptlib

# Load COPT shared library
coptlib = COPT_DLL_loadlib()

# COPT API name map
COPT_CreateEnv       = coptlib.COPT_CreateEnv
COPT_DeleteEnv       = coptlib.COPT_DeleteEnv
COPT_CreateProb      = coptlib.COPT_CreateProb
COPT_DeleteProb      = coptlib.COPT_DeleteProb
COPT_LoadProb        = coptlib.COPT_LoadProb
COPT_AddCols         = coptlib.COPT_AddCols
COPT_WriteMps        = coptlib.COPT_WriteMps
COPT_WriteLp         = coptlib.COPT_WriteLp
COPT_WriteBin        = coptlib.COPT_WriteBin
COPT_WriteSol        = coptlib.COPT_WriteSol
COPT_WriteBasis      = coptlib.COPT_WriteBasis
COPT_WriteMst        = coptlib.COPT_WriteMst
COPT_WriteParam      = coptlib.COPT_WriteParam
COPT_AddMipStart     = coptlib.COPT_AddMipStart
COPT_SolveLp         = coptlib.COPT_SolveLp
COPT_Solve           = coptlib.COPT_Solve
COPT_GetSolution     = coptlib.COPT_GetSolution
COPT_GetLpSolution   = coptlib.COPT_GetLpSolution
COPT_GetIntParam     = coptlib.COPT_GetIntParam
COPT_SetIntParam     = coptlib.COPT_SetIntParam
COPT_GetDblParam     = coptlib.COPT_GetDblParam
COPT_SetDblParam     = coptlib.COPT_SetDblParam
COPT_GetIntAttr      = coptlib.COPT_GetIntAttr
COPT_GetDblAttr      = coptlib.COPT_GetDblAttr
COPT_SearchParamAttr = coptlib.COPT_SearchParamAttr
COPT_SetLogFile      = coptlib.COPT_SetLogFile

# COPT LP/MIP status map
coptlpstat = {0: LpStatusNotSolved,
              1: LpStatusOptimal,
              2: LpStatusInfeasible,
              3: LpStatusUnbounded,
              4: LpStatusNotSolved,
              5: LpStatusNotSolved,
              6: LpStatusNotSolved,
              8: LpStatusNotSolved,
              9: LpStatusNotSolved,
              10: LpStatusNotSolved}

# COPT variable types map
coptctype = {LpContinuous: coptstr('C'), 
             LpBinary: coptstr('B'), 
             LpInteger: coptstr('I')}

# COPT constraint types map
coptrsense = {LpConstraintEQ: coptstr('E'), 
              LpConstraintLE: coptstr('L'), 
              LpConstraintGE: coptstr('G')}

# COPT objective senses map
coptobjsen = {LpMinimize: 1, 
              LpMaximize: -1}


class COPT_DLL(LpSolver):
  """
  The COPT dynamic library solver
  """
  def __init__(self, mip=True, msg=True, mip_start=False, logfile=None, **params):
    """
    Initialize COPT solver
    """
    LpSolver.__init__(self, mip, msg)

    # Initialize COPT environment and problem
    self.coptenv  = None
    self.coptprob = None

    # Use MIP start information
    self.mipstart = mip_start

    # Create COPT environment and problem
    self.create()

    # Set log file
    if logfile is not None:
      rc = COPT_SetLogFile(self.coptprob, coptstr(logfile))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to set log file")

    # Set parameters to problem
    if not self.msg:
      self.setParam("Logging", 0)

    for parname, parval in params.items():
      self.setParam(parname, parval)

  def available(self):
    """
    True if dynamic library is available
    """
    return True
        
  def actualSolve(self, lp):
    """
    Solve a well formulated LP/MIP problem

    This function borrowed implementation of CPLEX_DLL.actualSolve,
    with some modifications.
    """
    # Extract problem data and load it into COPT
    ncol, nrow, nnonz, objsen, objconst, colcost, \
      colbeg, colcnt, colind, colval, \
      coltype, collb, colub, rowsense, rowrhs, \
      colname, rowname = self.extract(lp)

    rc = COPT_LoadProb(self.coptprob, ncol, nrow, objsen, objconst, colcost,
                       colbeg, colcnt, colind, colval, coltype, collb, colub,
                       rowsense, rowrhs, None, colname, rowname)
    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to load problem")
    
    if lp.isMIP() and self.mip:
      # Load MIP start information
      if self.mipstart:
        mstdict = {self.v2n[v]: v.value() for v in lp.variables() \
                                          if v.value() is not None}

        if mstdict:
          mstkeys = ctypesArrayFill(list(mstdict.keys()), ctypes.c_int)
          mstvals = ctypesArrayFill(list(mstdict.values()), ctypes.c_double)

          rc = COPT_AddMipStart(self.coptprob, len(mstkeys), mstkeys, mstvals)
          if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to add MIP start information")

      # Solve the problem
      rc = COPT_Solve(self.coptprob)
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to solve the MIP problem")
    elif lp.isMIP() and not self.mip:
      # Solve MIP as LP
      rc = COPT_SolveLp(self.coptprob)
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to solve MIP as LP")
    else:
      # Solve the LP problem
      rc = COPT_SolveLp(self.coptprob)
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to solve the LP problem")

    # Get problem status and solution
    status = self.getsolution(lp, ncol, nrow)

    # Reset attributes
    for var in lp.variables():
      var.modified = False

    return status

  def extract(self, lp):
    """
    Extract data from PuLP lp structure

    This function borrowed implementation of LpSolver.getCplexStyleArrays,
    with some modifications.
    """
    cols     = list(lp.variables())
    ncol     = len(cols)
    nrow     = len(lp.constraints)

    collb    = (ctypes.c_double * ncol)()
    colub    = (ctypes.c_double * ncol)()
    colcost  = (ctypes.c_double * ncol)()
    coltype  = (ctypes.c_char   * ncol)()
    colname  = (ctypes.c_char_p * ncol)()

    rowrhs   = (ctypes.c_double * nrow)()
    rowsense = (ctypes.c_char   * nrow)()
    rowname  = (ctypes.c_char_p * nrow)()

    spmat    = sparse.Matrix(list(range(nrow)), list(range(ncol)))

    # Objective sense and constant offset
    objsen   = coptobjsen[lp.sense]
    objconst = ctypes.c_double(0.0)

    # Associate each variable with a ordinal
    self.v2n       = dict(((cols[i], i) for i in range(ncol)))
    self.vname2n   = dict(((cols[i].name, i) for i in range(ncol)))
    self.n2v       = dict((i, cols[i]) for i in range(ncol))
    self.c2n       = {}
    self.n2c       = {}
    self.addedVars = ncol
    self.addedRows = nrow

    # Extract objective cost
    for col, val in lp.objective.items():
      colcost[self.v2n[col]] = val

    # Extract variable types, names and lower/upper bounds
    for col in lp.variables():
      colname[self.v2n[col]] = coptstr(col.name)

      if col.lowBound is not None:
        collb[self.v2n[col]] = col.lowBound
      else:
        collb[self.v2n[col]] = -1e30

      if col.upBound is not None:
        colub[self.v2n[col]] = col.upBound
      else:
        colub[self.v2n[col]] = 1e30

    # Extract column types
    if lp.isMIP():
      for var in lp.variables():
        coltype[self.v2n[var]] = coptctype[var.cat]
    else:
      coltype = None

    # Extract constraint rhs, senses and names
    idx = 0
    for row in lp.constraints:
      rowrhs[idx] = -lp.constraints[row].constant
      rowsense[idx] = coptrsense[lp.constraints[row].sense]
      rowname[idx] = coptstr(row)

      self.c2n[row] = idx
      self.n2c[idx] = row
      idx += 1

    # Extract coefficient matrix and generate CSC-format matrix
    for col, row, coeff in lp.coefficients():
      spmat.add(self.c2n[row], self.vname2n[col], coeff)

    nnonz, _colbeg, _colcnt, _colind, _colval = spmat.col_based_arrays()

    colbeg = ctypesArrayFill(_colbeg, ctypes.c_int)
    colcnt = ctypesArrayFill(_colcnt, ctypes.c_int)
    colind = ctypesArrayFill(_colind, ctypes.c_int)
    colval = ctypesArrayFill(_colval, ctypes.c_double)

    return ncol, nrow, nnonz, objsen, objconst, colcost, colbeg, colcnt, \
           colind, colval, coltype, collb, colub, rowsense, rowrhs, \
           colname, rowname

  def create(self):
    """
    Create COPT environment and problem

    This function borrowed implementation of CPLEX_DLL.grabLicense,
    with some modifications.
    """
    # In case recreate COPT environment and problem
    self.delete()

    self.coptenv  = ctypes.c_void_p()
    self.coptprob = ctypes.c_void_p()

    # Create COPT environment
    rc = COPT_CreateEnv(byref(self.coptenv))
    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to create environment")

    # Create COPT problem
    rc = COPT_CreateProb(self.coptenv, byref(self.coptprob))
    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to create problem")

  def __del__(self):
    """
    Destructor of COPT_DLL class
    """
    self.delete()

  def delete(self):
    """
    Release COPT problem and environment

    This function borrowed implementation of CPLEX_DLL.releaseLicense,
    with some modifications.
    """
    # Valid environment and problem exist
    if self.coptenv is not None and self.coptprob is not None:
      # Delete problem
      rc = COPT_DeleteProb(byref(self.coptprob))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to delete problem")

      # Delete environment
      rc = COPT_DeleteEnv(byref(self.coptenv))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to delete environment")

      # Reset to None
      self.coptenv  = None
      self.coptprob = None

  def getsolution(self, lp, ncols, nrows):
    """Get problem solution

    This function borrowed implementation of CPLEX_DLL.findSolutionValues,
    with some modifications.
    """
    status  = ctypes.c_int()
    x       = (ctypes.c_double * ncols)()
    dj      = (ctypes.c_double * ncols)()
    pi      = (ctypes.c_double * nrows)()
    slack   = (ctypes.c_double * nrows)()

    var_x     = {}
    var_dj    = {}
    con_pi    = {}
    con_slack = {}

    if lp.isMIP() and self.mip:
      hasmipsol = ctypes.c_int()
      # Get MIP problem satus
      rc = COPT_GetIntAttr(self.coptprob, coptstr("MipStatus"), byref(status))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to get MIP status")
      # Has MIP solution
      rc = COPT_GetIntAttr(self.coptprob, coptstr("HasMipSol"), byref(hasmipsol))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to check if MIP solution exists")

      # Optimal/Feasible MIP solution
      if status.value == 1 or hasmipsol.value == 1:
        rc = COPT_GetSolution(self.coptprob, byref(x))
        if rc != 0:
          raise PulpSolverError("COPT_PULP: Failed to get MIP solution")

        for i in range(ncols):
          var_x[self.n2v[i].name] = x[i]
        
      # Assign MIP solution to variables
      lp.assignVarsVals(var_x)
    else:
      # Get LP problem status
      rc = COPT_GetIntAttr(self.coptprob, coptstr("LpStatus"), byref(status))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to get LP status")

      # Optimal LP solution
      if status.value == 1:
        rc = COPT_GetLpSolution(self.coptprob, byref(x), byref(slack),
                                byref(pi), byref(dj))
        if rc != 0:
          raise PulpSolverError("COPT_PULP: Failed to get LP solution")

        for i in range(ncols):
          var_x[self.n2v[i].name] = x[i]
          var_dj[self.n2v[i].name] = dj[i]

        for i in range(nrows):
          con_pi[self.n2c[i]] = pi[i]
          con_slack[self.n2c[i]] = slack[i]

      # Assign LP solution to variables and constraints
      lp.assignVarsVals(var_x)
      lp.assignVarsDj(var_dj)
      lp.assignConsPi(con_pi)
      lp.assignConsSlack(con_slack)
    
    # Reset attributes
    lp.resolveOK = True
    for var in lp.variables():
      var.isModified = False

    lp.status = coptlpstat.get(status.value, LpStatusUndefined)
    return lp.status

  def write(self, filename):
    """
    Write problem, basis, parameter or solution to file
    """
    file_path = coptstr(filename)
    file_name, file_ext = os.path.splitext(file_path)

    if not file_ext:
      raise PulpSolverError("COPT_PULP: Failed to determine output file type")
    elif file_ext == coptstr('.mps'):
      rc = COPT_WriteMps(self.coptprob, file_path)
    elif file_ext == coptstr(".lp"):
      rc = COPT_WriteLp(self.coptprob, file_path)
    elif file_ext == coptstr(".bin"):
      rc = COPT_WriteBin(self.coptprob, file_path)
    elif file_ext == coptstr('.sol'):
      rc = COPT_WriteSol(self.coptprob, file_path)
    elif file_ext == coptstr('.bas'):
      rc = COPT_WriteBasis(self.coptprob, file_path)
    elif file_ext == coptstr('.mst'):
      rc = COPT_WriteMst(self.coptprob, file_path)
    elif file_ext == coptstr('.par'):
      rc = COPT_WriteParam(self.coptprob, file_path)
    else:
      raise PulpSolverError("COPT_PULP: Unsupported file type")

    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to write file '{}'".format(filename))

  def setParam(self, name, val):
    """
    Set parameter to COPT problem
    """
    par_type = ctypes.c_int()
    par_name = coptstr(name)

    rc = COPT_SearchParamAttr(self.coptprob, par_name, byref(par_type))
    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(par_name))

    if par_type.value == 0:
      rc = COPT_SetDblParam(self.coptprob, par_name, ctypes.c_double(val))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to set double parameter '{}'".format(par_name))
    elif par_type.value == 1:
      rc = COPT_SetIntParam(self.coptprob, par_name, ctypes.c_int(val))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to set integer parameter '{}'".format(par_name))
    else:
      raise PulpSolverError("COPT_PULP: Invalid parameter '{}'".format(par_name))

  def getParam(self, name):
    """
    Get current value of parameter
    """
    par_dblval = ctypes.c_double()
    par_intval = ctypes.c_int()
    par_type   = ctypes.c_int()
    par_name   = coptstr(name)

    rc = COPT_SearchParamAttr(self.coptprob, par_name, byref(par_type))
    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(par_name))

    if par_type.value == 0:
      rc = COPT_GetDblParam(self.coptprob, par_name, byref(par_dblval))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to get double parameter '{}'".format(par_name))
      else:
        retval = par_dblval.value
    elif par_type.value == 1:
      rc = COPT_GetIntParam(self.coptprob, par_name, byref(par_intval))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to get integer parameter '{}'".format(par_name))
      else:
        retval = par_intval.value
    else:
      raise PulpSolverError("COPT_PULP: Invalid parameter '{}'".format(par_name))

    return retval

  def getAttr(self, name):
    """
    Get attribute of the problem
    """
    attr_dblval = ctypes.c_double()
    attr_intval = ctypes.c_int()
    attr_type   = ctypes.c_int()
    attr_name   = coptstr(name)

    # Check attribute type by name
    rc = COPT_SearchParamAttr(self.coptprob, attr_name, byref(attr_type))
    if rc != 0:
      raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(attr_name))

    if attr_type.value == 2:
      rc = COPT_GetDblAttr(self.coptprob, attr_name, byref(attr_dblval))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to get double attribute '{}'".format(attr_name))
      else:
        retval = attr_dblval.value
    elif attr_type.value == 3:
      rc = COPT_GetIntAttr(self.coptprob, attr_name, byref(attr_intval))
      if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to get integer attribute '{}'".format(attr_name))
      else:
        retval = attr_intval.value
    else:
      raise PulpSolverError("COPT_PULP: Invalid attribute '{}'".format(attr_name))

    return retval
