using Multirate
using Winston

function wgn(n::Int; power::Real=1)
    return power * (randn(n) + randn(n)im)
end


function powerspectrum( x::Vector, window::Function = blackman )
    xLen = length( x )
    xWin = x .* window( xLen )
    10*log10.(fftshift(abs2.(fft( xWin ))))
end

function channelizerplots( signal::Vector, channelizedSignals::Matrix )
    (samplesPerChannel,Nchannels) = size( channelizedSignals )
    # Create plot tables to hold original spectrum plus spectrum and signal traces of each channel
    table              = Table(3,1)
    subTable           = Table(2,Nchannels)

    # Compute the spectrum of the original signal and add it to the table
    signalSpectrum     = powerspectrum( signal )
    p                  = plot( linspace(-0.5,0.5,length(signalSpectrum)), signalSpectrum )
    setattr( p, "title", "Original Signal" )
    table[1,1]         = p
    ylim(-10,60)
    
    table[2,1]         = plot( indices(signal,1), real(signal), "-b", indices(signal,1), imag(signal), "-r" )
    
    freqs = linspace( -0.5, 0.5, samplesPerChannel )
    t     = linspace( 0, Nchannels*samplesPerChannel-1, samplesPerChannel )
    
    # Plot the spectrum/time-domain traces of each channel and add them to the subtable
    for channel in 1:Nchannels
        thisSignal          = channelizedSignals[:,channel]
        spectrum            = powerspectrum( thisSignal )
        sp                  = plot( freqs, spectrum ); ylim(-10,60)
        setattr( sp, "title", "Channel $channel" )
        tp                  = plot( t, real(thisSignal), "-b", t, imag(thisSignal), "-r" )
        if Nchannels > 3
            setattr( sp.y1, "ticklabels", [])
            setattr( sp.x1, "ticklabels", [])
            setattr( tp.y1, "ticklabels", [])
            setattr( tp.x1, "ticklabels", [])
        end
        subTable[1,channel] = sp
        subTable[2,channel] = tp
    end

    # Position the subtable traces below the main spectrum trace
    table[3,1] = subTable
    
    return table
end

function segments(n; alpha=0.5)
    s0 = 1
    n0 = n
    segs = []
    
    while n > 0
        s=min(n, round(Int, alpha*rand()*n0))
        
        if s > 0
            push!(segs, s0:s0+s-1)
            n -= s
            s0 += s
        end
    end
    
    return segs
end

const Th = Float64
const Tx = Complex128 # Datatype for x
const ƒs = 1.0        # Input sample rate

const Nchannels = 35
const samplesPerChannel = 85000

# Construct a linear chirp signal from ƒ = -0.5 to -0.5
const n                   = Nchannels * samplesPerChannel
const t                   = 0:n-1

const ψ = π * (t / n - 1) .* t

const signal             = exp.(Array{Tx}(ψ) * 1im / ƒs) + wgn( n, power=0.1)

# Instantiate a channelizer with Nchannels
const channelizer        = Channelizer( Th, Tx, Nchannels, 32 )

# channelizedSignals = filt( channelizer, signal )
channelizedSignals = Array{Tx}(0, Nchannels)
for segment in segments(samplesPerChannel)
    expandedSegment = (segment.start-1)*Nchannels+1 : segment.stop*Nchannels
    @printf("Channelizing segment %s as %s\n", segment, expandedSegment)
    @time channelizedSignals = [channelizedSignals; filt( channelizer, signal[expandedSegment] )]
end

# Create the table of plots
table = channelizerplots( signal, channelizedSignals )

winOpen = [true]
win = Winston.window("Channelizer Example", 1500, 1000, path -> winOpen[1] = false)
display(win, table)

while(winOpen[1])
    sleep(0.1)
end
