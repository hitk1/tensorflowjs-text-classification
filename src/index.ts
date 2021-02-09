import * as tf from '@tensorflow/tfjs-node'
import * as use from '@tensorflow-models/universal-sentence-encoder'

(async () => {
    try {

        const universalEncoder = await use.load()
        const rawData = [
            'senhor bom é otimo e por isso é bom',
            'senhor ruim é pessimo e por isso é ruim', 
            'o bom é otimo porque é bom e pronto',
            'o ruim é pessimo porque é ruim e pronto'
        ]
        const xTrain = await universalEncoder.embed(rawData)

        const yTrain = tf.tensor2d([
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0]
        ])

        const model = tf.sequential()

        model.add(
            tf.layers.dense({
                inputShape: [512],
                activation: 'sigmoid',
                units: 2
            })
        )

        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: tf.train.adam(0.001),
            metrics: ['accuracy']
        })

        await model.fit(xTrain, yTrain, {
            epochs: 50
        })

        console.log('modelo treinado')

        //@ts-ignore
        const predicted = model.predict(await universalEncoder.embed('senhor bom')).print()

    } catch (error) {
        console.log(`Erro inesperado: ${error.message}`)
    }
})()    