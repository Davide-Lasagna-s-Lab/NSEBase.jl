@testset "_get_padded_shape                " begin
    # single transformed dimension
    @test NSEBase._get_padded_shape((4,), (1,)) == (7,)
    @test NSEBase._get_padded_shape((6,), (1,)) == (9,)
    @test NSEBase._get_padded_shape((8,), (1,)) == (13,)

    # untransformed dimensions are unchanged
    @test NSEBase._get_padded_shape((4, 5), (1,)) == (7, 5)
    @test NSEBase._get_padded_shape((4, 5), (2,)) == (4, 9)

    # multiple transformed dimensions
    @test NSEBase._get_padded_shape((4, 6), (1, 2))       == (7, 9)
    @test NSEBase._get_padded_shape((4, 6, 8), (1, 2, 3)) == (7, 9, 13)
    @test NSEBase._get_padded_shape((4, 5, 6), (1, 3))    == (7, 5, 9)

    # padded size must satisfy the 3/2 dealiasing rule for all transformed dimensions
    for s in 1:32, ndim in 1:3
        shape  = ntuple(_ -> s, ndim)
        order  = ntuple(identity, ndim)
        padded = NSEBase._get_padded_shape(shape, order)
        for d in order
            @test padded[d] >= 3s/2
        end
    end
end

@testset "_get_transform_shape             " begin
    # 1D: first (and only) dimension is halved + 1
    @test NSEBase._get_transform_shape((4,), 1) == (3,)
    @test NSEBase._get_transform_shape((6,), 1) == (4,)
    @test NSEBase._get_transform_shape((8,), 1) == (5,)

    # 2D: only the selected dimension changes
    @test NSEBase._get_transform_shape((4, 6), 1) == (3, 6)
    @test NSEBase._get_transform_shape((4, 6), 2) == (4, 4)

    # 3D
    @test NSEBase._get_transform_shape((4, 6, 8), 1) == (3, 6, 8)
    @test NSEBase._get_transform_shape((4, 6, 8), 2) == (4, 4, 8)
    @test NSEBase._get_transform_shape((4, 6, 8), 3) == (4, 6, 5)
end
