import Multirate: PFB, taps2pfb

# Interpolator FIR kernel
type Channelizer{Th,Tx}
    pfb::PFB{Th}
    h::Vector{Th}
    Nchannels::Int
    tapsPer𝜙::Int
    history::Array{Array{Tx}}
end

function Channelizer{Th}( Tx, h::Vector{Th}, Nchannels::Integer )
    pfb       = taps2pfb( h, Nchannels )
    Nchannels = size( pfb )[2]
    tapsPer𝜙  = size( pfb )[1]
    Channelizer{Th, Tx}( pfb, h, Nchannels, tapsPer𝜙, [] )
end

function Channelizer( Th, Tx, Nchannels::Integer, tapsPer𝜙 = 20 )
    hLen = tapsPer𝜙 * Nchannels
    h    = firdes( hLen, 0.45/Nchannels, kaiser ) .* Nchannels
    Channelizer( Tx, Array{Th}(h), Nchannels )
end




function filt!{Tb,Th,Tx}( buffer::Matrix{Tb}, kernel::Channelizer{Th}, x::AbstractVector{Tx} )
    Nchannels         = kernel.Nchannels
    pfb               = kernel.pfb
    tapsPer𝜙          = kernel.tapsPer𝜙
    xLen              = length( x )
    (bufLen,bufWidth) = size( buffer )
    fftBuffer         = Array{Tb}( Nchannels )

    @assert xLen   % Nchannels == 0
    @assert bufLen * bufWidth  == xLen
    @assert Tb                 == promote_type(Th,Tx)

    xPartitioned = Array{Vector{Tx}}( Nchannels )

    for channel in 1:Nchannels
        xIdxStart = Nchannels-channel+1
        xPartitioned[channel] = x[xIdxStart:Nchannels:end]
    end

    if kernel.history == []
        kernel.history = [ zeros(Tx, tapsPer𝜙-1) for i in 1:Nchannels ]
    end

    @simd for xIdx in 1:bufLen
        @simd for 𝜙Idx in Nchannels:-1:1
            if xIdx < tapsPer𝜙
                @inbounds fftBuffer[𝜙Idx] = unsafedot( pfb, 𝜙Idx, kernel.history[𝜙Idx], xPartitioned[𝜙Idx], xIdx )
            else
                @inbounds fftBuffer[𝜙Idx] = unsafedot( pfb, 𝜙Idx, xPartitioned[𝜙Idx], xIdx )
            end
        end

        @inbounds buffer[xIdx,:] = fftshift(ifft(fftBuffer))
    end

    # set history for next call
    @simd for 𝜙Idx in 1:Nchannels
        @inbounds kernel.history[𝜙Idx] = shiftin!( kernel.history[𝜙Idx], xPartitioned[𝜙Idx] )
    end

    return buffer
end

function filt{Th,Tx}( kernel::Channelizer{Th,Tx}, x::AbstractVector{Tx} )
    xLen   = length( x )

    @assert xLen % kernel.Nchannels == 0

    buffer = Array{promote_type(Th,Tx)}( Int(xLen/kernel.Nchannels), kernel.Nchannels )
    filt!( buffer, kernel, x )
    return buffer
end
