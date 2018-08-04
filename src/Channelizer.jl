import Multirate: PFB, taps2pfb

# Interpolator FIR kernel
type Channelizer{T}
    pfb::PFB{T}
    h::Vector{T}
    Nchannels::Int
    tapsPer𝜙::Int
    history::AbstractArray
end

function Channelizer( h::Vector, Nchannels::Integer )
    pfb       = taps2pfb( h, Nchannels )
    Nchannels = size( pfb )[2]
    tapsPer𝜙  = size( pfb )[1]
    Channelizer( pfb, h, Nchannels, tapsPer𝜙, [] )
end

function Channelizer( Nchannels::Integer, tapsPer𝜙 = 20 )
    hLen = tapsPer𝜙 * Nchannels
    h    = firdes( hLen, 0.45/Nchannels, kaiser ) .* Nchannels
    Channelizer( h, Nchannels )
end




function filt!{Tb,Th,Tx}( buffer::AbstractMatrix{Tb}, kernel::Channelizer{Th}, x::AbstractVector{Tx} )
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

    for xIdx in 1:bufLen
        for 𝜙Idx in Nchannels:-1:1
            if xIdx < tapsPer𝜙
                fftBuffer[𝜙Idx] = unsafedot( pfb, 𝜙Idx, kernel.history[𝜙Idx], xPartitioned[𝜙Idx], xIdx )
            else
                fftBuffer[𝜙Idx] = unsafedot( pfb, 𝜙Idx, xPartitioned[𝜙Idx], xIdx )
            end
        end

        buffer[xIdx,:] = fftshift(ifft(fftBuffer))
    end

    # set history for next call
    for 𝜙Idx in 1:Nchannels
        kernel.history[𝜙Idx] = shiftin!( kernel.history[𝜙Idx], xPartitioned[𝜙Idx] )
    end

    return buffer
end

function filt{Th,Tx}( kernel::Channelizer{Th}, x::AbstractVector{Tx} )
    xLen   = length( x )

    @assert xLen % kernel.Nchannels == 0

    buffer = Array{promote_type(Th,Tx)}( Int(xLen/kernel.Nchannels), kernel.Nchannels )
    filt!( buffer, kernel, x )
    return buffer
end
