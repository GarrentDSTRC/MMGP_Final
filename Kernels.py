import torch
import gpytorch
from gpytorch.kernels import Kernel
"-----------------------DIY KERNEL-----------------------------------------------"
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,mode="M"):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        if mode=="M":
        #self.covar_module =gpytorch.kernels.GridInterpolationKernel(
            #gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),grid_size=grid_size, num_dims=4)
            #self.covar_module = gpytorch.kernels.MaternKernel()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

        elif mode=="MR":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif mode=="R":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class SpectralMixtureGPModelBack(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModelBack, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=train_x.shape[1])
        self.covar_module.initialize_from_data(train_x, train_y)
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=train_x.shape[1], mixture_weights_constraint=gpytorch.constraints.Interval(-10, 10))
        # 设置mixture_weights参数的约束条件为-1到1之间
        self.covar_module.mixture_weights_constraint = gpytorch.constraints.Interval(-10, 10)
        # 初始化mixture_weights参数的值为-0.5
        self.covar_module.mixture_weights = -0.5

        self.covar_module.initialize_from_data(train_x, train_y)
        # 创建引导点
        inducing_points = train_x[:10, :]
        # 创建变分分布
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        # 创建变分策略，并增加jitter参数
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        # 把变分策略作为一个属性
        variational_strategy.jitter_val = 1e-3
        self.variational_strategy = variational_strategy
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class MultitaskGPModel(gpytorch.models.ExactGP):
    #model with output-covariance
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )

        # grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=2)
        #     , num_tasks=2, rank=1
        # )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
             gpytorch.kernels.RBFKernel()
             , num_tasks=2, rank=1
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
class TwoFidelityIndexKernel(Kernel):
    """
    Separate kernel for each task based on the Hadamard Product between the task
    kernel and the data kernel. based on :
    https://github.com/cornellius-gp/gpytorch/blob/master/examples/03_Multitask_GP_Regression/Hadamard_Multitask_GP_Regression.ipynb

    The index identifier must start from 0, i.e. all task zero have index identifier 0 and so on.

    If noParams is set to `True` then the covar_factor doesn't include any parameters.
    This is needed to construct the 2nd matrix in the sum, as in (https://arxiv.org/pdf/1604.07484.pdf eq. 3.2)
    where the kernel is treated as a sum of two kernels.

    k = [      k1, rho   * k1   + [0, 0
         rho * k1, rho^2 * k1]     0, k2]
    """

    def __init__(self,
                 num_tasks,
                 rank=1,  # for two multifidelity always assumed to be 1
                 prior=None,
                 includeParams=True,
                 **kwargs
                 ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)
        try:
            self.batch_shape
        except AttributeError as e:
            self.batch_shape = 1  # torch.Size([200])

        # we take a power of rho with the task index list (assuming all task 0 represented as 0, task 1 represented as 1 etc.)
        self.covar_factor = torch.arange(num_tasks).to(torch.float32)

        if includeParams:
            self.register_parameter(name="rho", parameter=torch.nn.Parameter(torch.randn(1)))
            print(f"Initial value : rho  {self.rho.item()}")
            self.covar_factor = torch.pow(self.rho.repeat(num_tasks), self.covar_factor)

        self.covar_factor = self.covar_factor.unsqueeze(-1)
        #self.covar_factor = self.covar_factor.repeat(self.batch_shape, 1, 1)

        if prior is not None and includeParams is True:
            self.register_prior("rho_prior", prior, self._rho)

    def _rho(self):
        return self.rho

    def _eval_covar_matrix(self):
        transp = self.covar_factor.transpose(-1, 0)
        ret = self.covar_factor.matmul(self.covar_factor.transpose(-1, -2))  # + D
        return ret

    @property
    def covar_matrix(self):
        res = RootLinearOperator(self.covar_factor)
        print("root",res.to_dense())
        return res

    def forward(self, i1, i2, **params):
        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = torch.broadcast_shapes(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLinearOperator(
            base_linear_op=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class MultiFidelityGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultiFidelityGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # self.covar_module1 = gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.MaternKernel()
        #     )
        # self.covar_module2 = gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.MaternKernel()
        #     )
        self.covar_module1 =  gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=int(train_x[0].shape[1]))
        self.covar_module1.initialize_from_data(train_x[0], train_y)
        self.covar_module2 =  gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=int(train_x[0].shape[1]))
        self.covar_module2.initialize_from_data(train_x[0], train_y)

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        # self.task_covar_module = IndexKernel(num_tasks=2, rank=1)
        self.task_covar_module1 = TwoFidelityIndexKernel(num_tasks=2, rank=1)
        self.task_covar_module2 = TwoFidelityIndexKernel(num_tasks=2, rank=1,
                                                         includeParams=False)  # , batch_shape=(train_y.shape[0],1,1))
        #self.task_covar_module1 = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    #         print(self.covar_module1.outputscale.item())
    #         print(self.covar_module1.base_kernel.lengthscale.item())
    #         pprint(dir(self.covar_module1))
    #         pprint(dir(self.covar_module1.base_kernel))

    # print(f"Initial value : Covar 1, lengthscale {self.covar_module1.base_kernel.lengthscale.item()}, prefactor {self.covar_module1.outputscale.item()}")
    # print(f"Initial value : Covar 2, lengthscale {self.covar_module2.base_kernel.lengthscale.item()}, prefactor {self.covar_module2.outputscale.item()}")

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar1_x = self.covar_module1(x)
        # Get task-task covariance
        covar1_i = self.task_covar_module1(i)

        # Get input-input covariance
        covar2_x = self.covar_module2(x)
        # Get task-task covariance
        covar2_i = self.task_covar_module2(i)

        # Multiply the two together to get the covariance we want
        covar1 = covar1_x.mul(covar1_i)
        covar2 = covar2_x.mul(covar2_i)
        #         covar1 = covar1_x * covar1_i
        #         covar2 = covar2_x * covar2_i pipreqs ./ --encoding=utf8

        return gpytorch.distributions.MultivariateNormal(mean_x, covar1 + covar2)

