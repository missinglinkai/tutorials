def do_solve(niter, solver, disp_interval, test_interval, test_iters, id):

    import tempfile
    import numpy as np
    import os
    # from pylab import zeros, arange, subplots, plt, savefig

    # SET PLOTS DATA
    # train_loss = zeros(niter/disp_interval)
    # train_acc = zeros(niter/disp_interval)
    # val_acc = zeros(niter/test_interval)
    #
    # it_axes = (arange(niter) * disp_interval) + disp_interval
    # it_val_axes = (arange(niter) * test_interval) + test_interval
    #
    # _, ax1 = subplots()
    # ax2 = ax1.twinx()
    # ax1.set_xlabel('iteration')
    # ax1.set_ylabel('train loss (r)')
    # ax2.set_ylabel('train accuracy (b), val accuracy (g)')
    # ax2.set_autoscaley_on(False)
    # ax2.set_ylim([0, 1])
    #
    # blobs = ('loss', 'acc')
    # loss, acc = (np.zeros(niter) for _ in blobs)


    #RUN TRAINING
    for it in range(niter):
        solver.step(1)  # run a single SGD step in Caffe
        # loss[it], acc[it] = (solver.net.blobs[b].data.copy() for b in blobs)

        #PLOT
        # if it % disp_interval == 0 or it + 1 == niter:
        #     loss_disp = 'loss=%.3f, acc=%2d%%' % (loss[it], np.round(100*acc[it]))
        #     print '%3d) %s' % (it, loss_disp)
        #
        #     train_loss[it/disp_interval - 1] = loss[it]
        #     train_acc[it/disp_interval - 1] = acc[it]
        #
        #     ax1.plot(it_axes[0:it/disp_interval], train_loss[0:it/disp_interval], 'r')
        #     ax2.plot(it_axes[0:it/disp_interval], train_acc[0:it/disp_interval], 'b')
        #     plt.ion()
        #     plt.show()
        #     plt.pause(0.001)

        #VALIDATE
        if it % test_interval == 0 and it > 0:
            accuracy = 0
            for i in range(test_iters):
                solver.test_nets[0].forward()
                accuracy += solver.test_nets[0].blobs['acc'].data
            accuracy /= test_iters
            print("Test Accuracy: {:.3f}".format(accuracy))

            # val_acc[it/test_interval - 1] = accuracy
            # ax2.plot(it_val_axes[0:it/test_interval], val_acc[0:it/test_interval], 'g')
            # plt.ion()
            # plt.title(id)
            # plt.show()
            # plt.pause(0.001)

            # Save training plot
            # title = '../../datasets/102flowers/training/training-' + id + '_' + str(it) + '.png'  # Save graph to disk
            # savefig(title, bbox_inches='tight')
            #
            # Save training data
            # outfile = '../../datasets/102flowers/training_data/' + id + '.txt'
            # np.savez(outfile, train_loss, train_acc, val_acc)




    return