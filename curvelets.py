#import numpy as np
import cupy as cp
from ArraysCollection import ArraysCollection
import functools
import math 
class DirectionalFilterBank():
    """ Class that perform a directional filter bank, which means one step of the curvelet transform for one scale.
        The procedure was taken from the article the uniform discrete curvelet transform.
    """
    def _compute_decimation_factor_from_n_angles(self, n_angles):
        """ Fonction that compute the decimation factor i.e. the factor that will divide horizontally the vertical angular
            window, and divide vertically the horizontal angular window.

        Args:
            n_angles (Integer): number of angular window per direction (one direction means horizontal or vertical)

        Returns:
            Integer: the decimation factor that will divide horizontally  the vertical angular window, and divide vertically the
                     horizontal angular window
        """
        self.decimation_factor = 1
        while self.decimation_factor * 2 < n_angles:
            self.decimation_factor *= 2
        return self.decimation_factor
    
    def T_angle(self,x,y):
        """ Elementwise function that, given x and y pixel coordinates return T(theta(x,y)) with T given in the article

        Args:
            x (array): coordinate along x
            y (array): coordinate along y

        Returns:
            [array]: same dimension of x and y. For each element xi, yi of x and y returns T(theta(xi,yi)).
        """
        result = cp.zeros(x.shape)
        result = cp.where(x >= abs(y), y/(x+ 1e-18), result)
        result = cp.where(y >= abs(x), 2 - x/(y+ 1e-18), result)
        result = cp.where(y <= - abs(x), -2 - x/(y+ 1e-18), result)
        result = cp.where(x <=  - abs(y),   (y>=0) * (  4 + y/(x+ 1e-18)) \
                                         + (y< 0) * ( -4 + y/(x+ 1e-18))
                                       , result
                         )
        result = cp.where(cp.logical_and(x == 0, y == 0), 0, result)
        return result
    
    def _get_frame_functions(self, n_angles, nu_a = 0.3, nu_b = 0.2):
        """ Function that setup the frame function given in the article. The name are the same

        Args:
            n_angles (Integer):  number of angular window per direction (one direction means horizontal or vertical)
            nu_a (float, optional): smoothing factor for concentric windows. Defaults to 0.3.
            nu_b (float, optional): smoothing factor for angular windows. Defaults to 0.2.

        Returns:
            function, function: the windows function corresponding to the low pass filter and the angular filters 
        """
        poly = lambda t: 1/2 + t * (35/32 + t**2 * (-35/32 + t**2 * (21/32 - 5/32 * t**2)))
        #cp.polyval([- 5/32, 0, 21/32, 0, -35/32, 0, 35/32,1/2], t)
        beta_squared = lambda t: cp.where(cp.abs(t) < 1, poly(t), (t > 0).astype(float))
        safe_beta_squared = lambda t: cp.clip(beta_squared(t), 0, 1) # when rounding error makes values out of [0,1] 
        beta = lambda t : cp.sqrt(safe_beta_squared(t))
        w1_tilda = lambda t : beta((1 - abs(t))/nu_a)
        w0_tilda = lambda t : w1_tilda((2 * (1 + nu_a)) * t)
        w0 = lambda cx, cy : w0_tilda(cx)* w0_tilda(cy)
        w1 = lambda cx, cy : cp.sqrt(1 - w0(cx, cy) ** 2) * w1_tilda(cx) * w1_tilda(cy)
        
        width_angle = 2/n_angles
        denominator = width_angle * nu_b
        v1_tilda = lambda t:   beta( ((width_angle - 1) - t)/ denominator ) \
                             * beta(    ( t + 1 )           / denominator )
        v_tilda  = lambda t, idx_angle : v1_tilda( t - width_angle * idx_angle )
        
        v = lambda cx,cy, idx_angle : v_tilda(self.T_angle(cx,cy), idx_angle)
        
        u_tilda = lambda cx, cy, idx_angle:  w1(cx, cy) * v(cx,cy, idx_angle)
        
        self.beta = beta
        self.w0 = w0
        self.w1 = w1
        self.v_tilda = v_tilda
        self.v = v
        self.u_tilda = u_tilda
        return self.w0, self.u_tilda
    
    def _compute_angular_filters(self,size_image, n_angles, border):
        """ Function that precompute the filters that will be used for the filter bank

        Args:
            size_image (Integer): the size of the image that will be given as input to the transform
            n_angles (Integer): number of angular window per direction (one direction means horizontal or vertical)
            border (str): "null" or "toric". Depending on the hypothesis made on extrapolation outside borders.

        Returns:
            array: a 3D dimensional array, which is astack of images each corresponding to one filters.
        """
        graduation = cp.arange(- size_image // 2, size_image // 2)
        x,y = cp.meshgrid(graduation, graduation, indexing = 'ij')
        x = x / (size_image // 2)
        y = y / (size_image // 2)
        x = cp.fft.fftshift(x)
        y = cp.fft.fftshift(y)
        self.lowpass_filter  = cp.expand_dims(self.w0(x,y), axis = 0)
        
        if border == "toric":
            # self.angular_filters = cp.array( [ [ [
            #     self.u_tilda(x + px,y + py, idx_angle) for px in [-2,0,2]
            #                                      ]     for py in [-2,0,2]
            #                                    ]       for idx_angle in range(n_angles*2)
            #                                  ]
            #                                )
            ang_frame_func = lambda idx_angle: cp.sum(
                cp.array(
                    [ 
                        [ self.u_tilda(x + px,y + py, idx_angle) for px in [-2,0,2] ]
                                                            for py in [-2,0,2]
                    ]
                ),
                axis = (0,1)
            )
            self.angular_filters = cp.array( 
                [ 
                    ang_frame_func(idx_angle)
                    for idx_angle in range(n_angles*2)
                ]
            )
            
        elif border == "null":
            self.angular_filters = cp.array( [ self.u_tilda(x ,y , idx_angle)
                                               for idx_angle in range(n_angles*2)
                                             ]
                                           )
        self.filters = cp.concatenate( ( self.angular_filters, self.lowpass_filter ), axis = 0 )
        return self.filters
    
    def __init__(self, size_image, n_angles, nu_a = 0.3, nu_b = 0.2, border="null"):
        """ init the transform with its hyper parameters
        Args:
            size_image (Integer): the size of the image that will be given as input to the transform
            n_angles (Integer): number of angular window per direction (one direction means horizontal or vertical)
            nu_a (float, optional): smoothing factor for concentric windows. Defaults to 0.3.
            nu_b (float, optional): smoothing factor for angular windows. Defaults to 0.2.
            border (str): "null" or "toric". Depending on the hypothesis made on extrapolation outside borders.
        """
      
        self.n_angles = n_angles
        self.nu_a = nu_a
        self.nu_b = nu_b
        self.border = border
        self._compute_decimation_factor_from_n_angles(n_angles) 
        self._get_frame_functions(n_angles, nu_a = 0.3, nu_b = 0.2)
        self._compute_angular_filters(size_image, n_angles, border)

        
    def _decimation(self,arr, coef, axis):
        """ Function that performs a time/spatial decimation in the frequency domain

        Args:
            arr (array): input data
            coef (Integer): decimation factor
            axis (Integer): axis that determines the direction of decimation

        Returns:
            array: the new decimated array
        """

        return functools.reduce( lambda a,b : a+b, 
                                 cp.split( arr  , 
                                           coef , 
                                           axis = axis
                                         )
                               )
        
    def __call__(self,image):
        """ Function that performs the directional filter bank of the image as it is described in the article. This function can be applied 
            to a batch of images. This batch can have any shape. In this case the outputs will bet a set of batch too. 

        Args:
            image (array): input data of dimension (n1 x n2 x ... ) x n x n. n1 x ... are the dimension of the batch. and n is the size of the square image

        Returns:
            tuple : Three arrays:
                        - (n1 x n2 x ... ) x n_angle x 2 x (n/2) x (n/2) : the low frequencies of the directional filter bank
                        - (n1 x n2 x ... ) x n_anglcp.get_default_memory_pool()e x 2 x nk x (n/2) : the vertical frequencies of the directional filter bank
                        - (n1 x n2 x ... ) x n_angle x 2 x (n/2) x nk : the horizontal frequencies of the directional filter bank
        """

        fft = cp.fft.fft2(image, norm = "ortho")
        ndims_image = len(fft.shape)
        ndims_filter = 3
        axis_filter = ndims_image - 2
        axis_real_imag = axis_filter + 1
        
        expanded_filters = self.filters
        for _ in range(axis_filter):
            expanded_filters = cp.expand_dims(expanded_filters, axis = 0)
        fft = cp.expand_dims(fft, axis = axis_filter)
        
        filtered_fft = fft * expanded_filters
        
        filtered_fft = cp.expand_dims( filtered_fft, axis_real_imag )
        
        
        vdirectional_filtered, hdirectional_filtered, lowfreq_filtered  = \
                cp.split(     filtered_fft, 
                              [self.n_angles, 2* self.n_angles], 
                              axis = axis_filter 
                        )
        lowfreq_filtered = self._decimation(lowfreq_filtered, 2 , -1)
        lowfreq_filtered = self._decimation(lowfreq_filtered, 2 , -2)
        vdirectional_filtered = self._decimation(vdirectional_filtered, 2, -2)
        vdirectional_filtered = self._decimation(vdirectional_filtered, self.decimation_factor , -1)
        hdirectional_filtered = self._decimation(hdirectional_filtered, self.decimation_factor , -2)
        hdirectional_filtered = self._decimation(hdirectional_filtered, 2 , -1)
        
        hdirectional_filtered = cp.fft.ifft2(hdirectional_filtered, norm = "ortho")
        vdirectional_filtered = cp.fft.ifft2(vdirectional_filtered, norm = "ortho")
        lowfreq_filtered = cp.fft.ifft2(lowfreq_filtered, norm = "ortho")
        
        hdirectional_filtered = cp.concatenate( ( hdirectional_filtered.real, 
                                                  hdirectional_filtered.imag
                                                ), 
                                                axis = axis_real_imag
                                              )
        vdirectional_filtered = cp.concatenate( ( vdirectional_filtered.real, 
                                                  vdirectional_filtered.imag
                                                ), 
                                                axis = axis_real_imag
                                              )
        
        hdirectional_filtered = hdirectional_filtered * math.sqrt(2)
        vdirectional_filtered = vdirectional_filtered * math.sqrt(2)
        lowfreq_filtered = lowfreq_filtered.real
        
        return (lowfreq_filtered, vdirectional_filtered, hdirectional_filtered)
    
    def reconstruction(self, lowfreq_filtered, vdirectional_filtered, hdirectional_filtered):
        """ Function that performs the inverse directional filter bank of a transform as it is described in the article. This function can be applied 
            to a batch of transforms. This batch can have any shape. In this case the outputs will bet a set of batch too. 

        Args:
            lowfreq_filtered ([array]): (n1 x n2 x ... ) x 1 x 1 x (n/2) x (n/2) : the low frequencies of the directional filter bank
            vdirectional_filtered ([array]): (n1 x n2 x ... ) x n_angle x 2 x nk x (n/2) : the vertical frequencies of the directional filter bank
            hdirectional_filtered ([array]): (n1 x n2 x ... ) x n_angle x 2 x (n/2) x nk : the horizontal frequencies of the directional filter bank

        Returns:
            [array]: reconstructed image from the transform of size (n1 x n2 x ... ) x n x n 
        """
                 
        ndims_image = len(lowfreq_filtered.shape) - 2
        axis_filter = ndims_image - 2
        axis_real_imag = axis_filter + 1
        
        expanded_filters = self.filters
        for _ in range(axis_filter):
            expanded_filters = cp.expand_dims(expanded_filters, axis = 0)
        
        get_real_part = lambda arr: cp.take(arr, 0, axis = axis_real_imag)
        get_imag_part = lambda arr: cp.take(arr, 1, axis = axis_real_imag)
        to_complex    = lambda arr: get_real_part(arr) + 1j * get_imag_part(arr)
        
        
        lowfreq_filtered = cp.fft.fft2(lowfreq_filtered, norm = "ortho")
        lowfreq_filtered = cp.squeeze(lowfreq_filtered, axis = axis_real_imag)
        
        
        hdirectional_filtered = cp.fft.fft2(  to_complex(hdirectional_filtered), norm = "ortho" ) /math.sqrt(2)
        
        vdirectional_filtered = cp.fft.fft2(  to_complex(vdirectional_filtered), norm = "ortho") /math.sqrt(2)
        
        lowfreq_filtered = cp.tile(lowfreq_filtered, [1] * (ndims_image - 1) + [2,2]) 
        hdirectional_filtered = cp.tile( hdirectional_filtered, [1] * (ndims_image - 1) + [self.decimation_factor,2] )
        vdirectional_filtered = cp.tile( vdirectional_filtered, [1] * (ndims_image - 1) + [2,self.decimation_factor] )
        
        filtered_fft = cp.concatenate((vdirectional_filtered, hdirectional_filtered, lowfreq_filtered), axis = axis_filter)
        filtered_fft = filtered_fft * expanded_filters
        
        hf_filtered, lowfreq_filtered = cp.split(filtered_fft, [2*self.n_angles], axis = axis_filter)
        lowfreq_filtered = cp.squeeze(lowfreq_filtered, axis = axis_filter)
        hf_filtered =  cp.sum( hf_filtered, axis = axis_filter)
        
        
        hf_filtered_flipped = cp.flip(hf_filtered, axis =(-1))
        hf_filtered_flipped = cp.roll(hf_filtered_flipped, 1, axis =(-1))
        hf_filtered_flipped = cp.flip(hf_filtered_flipped, axis =(-2))
        hf_filtered_flipped = cp.roll(hf_filtered_flipped, 1, axis =(-2))
        

        hf_filtered = hf_filtered + cp.conj(hf_filtered_flipped)
        return cp.fft.ifft2(hf_filtered + lowfreq_filtered, norm = "ortho").real
 
class CurveletsOperator():
    """ Class that compute the curvelet transform. The procedure was taken from the article the uniform discrete curvelet transform.
    """
    def __init__(self, size_image, nums_angles, nu_a = 0.3, nu_b = 0.2):
        """ init the transform with its hyper parameters
        Args:
            size_image (Integer): the size of the image that will be given as input to the transform
            nums_angles (List[Integer]): number of angular window per direction (one direction means horizontal or vertical)
                                         given for each scale, from the coarsest to the finest
            nu_a (float, optional): smoothing factor for concentric windows. Defaults to 0.3.
            nu_b (float, optional): smoothing factor for angular windows. Defaults to 0.2.
        """
        self._directional_filter_banks = []
        border = "toric"
        size = size_image
        self.nums_angles = list(nums_angles)
        for num_angle in reversed(self.nums_angles):
            self._directional_filter_banks += [DirectionalFilterBank(size, num_angle, nu_a, nu_b, border)]
            size = size/2
            border="null"
        
    
    def __call__(self, image):
        """
        Compute the curvelet transform from the image
        Args:
            image (array):  a batch of images of size (n1 x n2 x ... ) x n x n 

        Returns:
            [List(array)]: the list of elements of the transform u0, the lowest frequencies, and u_js, the angular windows, were j is the scale, 
                           s the direction. each of its element have a size of (n1 x n2 x ... ) x na x nb x nc x nd  with na, nb, nc and nd vary
                           from one element ton another. 
        """
        result = [cp.expand_dims(image, axis = (-4,-3))]
        for dir_filt_bank in self._directional_filter_banks:
            result = list(dir_filt_bank(cp.squeeze(result[0], axis = (-4,-3)))) + result[1:]
        return ArraysCollection(result)
    
    def inverse(self, transform):
        """
            Compute the inverse curvelet transform
        Args:
            transform (List(array)): The orignal transform. Must not be build from scratch. usually a result of the __call__ method.
                                     the list of elements of the transform u0, the lowest frequencies, and u_js, the angular windows, were j is the scale, 
                                     s the direction. each of its element have a size of (n1 x n2 x ... ) x na x nb x nc x nd  with na, nb, nc and nd vary
                                     from one element ton another. (n1 x n2 x ... ) is the size of the batch.

        Returns:
            image (array):  a batch of images of size (n1 x n2 x ... ) x n x n
        """
        result = transform
        for dir_filt_bank in reversed(self._directional_filter_banks):
            result = [dir_filt_bank.reconstruction(result[0], result[1], result[2])] + list(result[3:])
            result[0] = cp.expand_dims(result[0], axis = (-4,-3))
        return cp.squeeze(result[0], axis = (-4,-3))
        
        
        
    
    
            
    
        